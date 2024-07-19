
import torch
import os
import argparse
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

from torch.utils.data import DataLoader

from eval.vilfw import evaluate
from lr_scheduler import PolynomialLRWarmup
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import init_logging,AverageMeter
from torch.utils.tensorboard import SummaryWriter
from datetime import timedelta
from datetime import datetime
from dataset import get_dataloader
import  numpy as np
import wandb
from ghostfacenetsv2 import GhostFaceNetsV2
from losses import  CombinedMarginLoss
from margin_model import MarginModel
import logging
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from torch.nn import  CrossEntropyLoss
from eval import vilfw
import pandas as pd
from uploadggdr import uploadDrive
#GPU memory
# torch.cuda.empty_cache()
# for key in os.environ.keys():
#     print(key)
IMAGE_SIZE = 112

rank = 0
local_rank = 0
world_size = 1



def main(cfg):
    # get config
    # cfg = get_config(args.config)
    # gauth = GoogleAuth()
    # drive = GoogleDrive(gauth)
    print("line 31: config: {}".format(cfg))
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)
    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    )
    # use wandb
    wandb_logger = None
    if cfg.using_wandb:

        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"

        try:
            if cfg.wandb_resume_status:
                wandb_logger = wandb.init(
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    sync_tensorboard=True,
                    resume="must",
                    id=cfg.wandb_id
                ) if rank == 0 or cfg.wandb_log_all else None
            else:
                wandb_logger = wandb.init(
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    sync_tensorboard=True,
                    name=run_name,
                    resume= False
                    ) if rank == 0 or cfg.wandb_log_all else None

                if wandb_logger:
                    wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")

    train_loader = get_dataloader(
        cfg.rec,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    # backbone = get_model(
    #     cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    backbone = GhostFaceNetsV2(image_size=IMAGE_SIZE,  width=1.3, dropout=0.2).cuda()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    backbone.train()
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    criterion =CrossEntropyLoss()
    margin_model =  MarginModel(embedding=512, numclass= cfg.num_classes)
    margin_model.cuda()
    margin_model.train()

    opt = torch.optim.SGD(
    params=[{"params": backbone.parameters()}, {"params": margin_model.parameters()}],
    lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)
    print("lr_scheduler",lr_scheduler)
    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_{cfg.checkpoint}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        margin_model.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))
    highest_acc = 0

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )
    loss_am = AverageMeter()
    all_img = pd.read_csv(cfg.pair_path)
    dataset = vilfw.VILWFDataset(cfg.data_dir,all_img)
    val_loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=4, shuffle=False)
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    callback_verification = CallBackVerification(
        summary_writer=summary_writer,
        wandb_logger = wandb_logger, bath_size = cfg.batch_size,
        val_loader=val_loader
    )
    for epoch in range(start_epoch, cfg.num_epoch):
        for _, (img, local_labels) in enumerate(train_loader):
            img =img.cuda()
            local_labels = local_labels.cuda()
            global_step += 1
            local_embeddings = backbone(img)
            local_embeddings = margin_model(local_embeddings)
            logits = margin_loss(local_embeddings, local_labels)
            loss = criterion(logits,local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()
            with torch.no_grad():
                if wandb_logger:
                    loss_am.update(loss.item(), 1)
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch,
                        'Learning rate/Step': opt.param_groups[0]["lr"]
                    })


                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:

                    callback_verification(global_step, backbone)
                    if epoch > 0.9*cfg.num_epoch :
                        if highest_acc >loss_am.avg:
                            checkpoint = {
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "state_dict_backbone": backbone.state_dict(),
                                "state_dict_softmax_fc": margin_model.state_dict(),
                                "state_optimizer": opt.state_dict(),
                                "state_lr_scheduler": lr_scheduler.state_dict(),
                                "author": "namkuner"
                            }
                            torch.save(checkpoint, os.path.join(cfg.output, "best.pt"))
                            # ckpt =[]
                            # ckpt.append(os.path.join(cfg.output, "best.pt") )
                            # uploadDrive(drive,ckpt)




        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.state_dict(),
                "state_dict_softmax_fc": margin_model.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
                "author" : "namkuner"
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_{epoch}.pt"))
            # ckpt = []
            # ckpt.append(os.path.join(cfg.output, f"checkpoint_{epoch}.pt"))
            # uploadDrive(drive, ckpt)

        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.state_dict(), path_module)

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"GhostFaceNets_E{epoch}"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)
        if cfg.dali:
            train_loader.reset()


    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.state_dict(), path_module)

        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"GhostFaceNets_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    # parser.add_argument("config", type=str, help="py config file")





    from configs.ghostfacenets_resume import config
    main(config)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # backbone = GhostFaceNetsV2(image_size=112, width=1.3, dropout=0.1).cuda()
    # backbone.eval()
    # same_acc, diff_acc, overall_acc, auc,threshs=evaluate("VILFWCut", "eval/output.csv", backbone, 2)
    # print(same_acc,diff_acc,overall_acc,auc)