# from dataset import get_dataloader
# from configs.ghostfacenets import config as cfg
# def main():
#
#     local_rank = 0
#     train_loader = get_dataloader(
#         cfg.rec,
#         local_rank,
#         cfg.batch_size,
#         cfg.dali,
#         cfg.dali_aug,
#         cfg.seed,
#         cfg.num_workers
#     )
#     max = 0
#     for batch_index,(data, label) in enumerate(train_loader):
#         if batch_index <= 4:
#             print(f"Batch {batch_index}")
#             print(f"Batch shape {data.shape}")
#             print(f"Data {data}")
#             print(f"Label {label}")
#             print(f"first data {data[0]}")
#         max +=1
#
#     print(f"Num of batches {max}")


def main():
    import torch
    from ghostfacenetsv2 import GhostFaceNetsV2
    local_rank  =0
    backbone = GhostFaceNetsV2(image_size=112, num_classes=10, width=1, dropout=0.).cuda()


    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    backbone.train()
def t():
    import torch;
    a = torch.ones(1, device="cuda")
if __name__ =='__main__' :
    # def count_classes(file_path):
    #     class_set = set()
    #
    #     with open(file_path, 'r') as file:
    #         for line in file:
    #             parts = line.strip().split()
    #             if len(parts) < 2:
    #                 continue  # Bỏ qua dòng không hợp lệ
    #             class_id = parts[1]
    #             print(parts[1])
    #             class_set.add(class_id)
    #     print(class_set)
    #     return len(class_set)
    #
    #
    # # Thay 'train.lst' bằng đường dẫn tới file của bạn
    # file_path = 'small_dataset/train.lst'
    # total_classes = count_classes(file_path)
    # print(f'Tổng số lớp: {total_classes}')
    import wandb
    from configs.ghostfacenets import config as cfg
    from datetime import datetime
    rank =0
    try:
        wandb.login(key=cfg.wandb_key)
    except Exception as e:
        print("WandB Key must be provided in config file (base.py).")
        print(f"Config Error: {e}")
    run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
    run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"

    if cfg.wandb_resume_status:
        wandb_logger = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            sync_tensorboard=True,
            resume="must",
            id=cfg.wandb_id
        ) if rank == 0 or cfg.wandb_log_all else None
        if wandb_logger:
            wandb_logger.config.update(cfg)

    else:
        wandb_logger = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            sync_tensorboard=True,
            name=run_name,
            resume=False
        ) if rank == 0 or cfg.wandb_log_all else None

        if wandb_logger:
            wandb_logger.config.update(cfg,allow_val_change=True)
