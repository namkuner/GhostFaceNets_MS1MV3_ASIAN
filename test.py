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
    pass