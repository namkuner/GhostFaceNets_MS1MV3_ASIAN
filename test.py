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
    # print(f'Tổng số lớp: {total_classes}')import os
    import os
    # Đường dẫn đến thư mục chứa các thư mục con cần đổi tên
    parent_dir = 'VILFWCut'

    # Lấy danh sách các thư mục con
    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    # Sắp xếp danh sách các thư mục con
    subdirs.sort()
    print(subdirs)


    for i, subdir in enumerate(subdirs):
        new_name = str(i + 1)
        old_path = os.path.join(parent_dir, subdir)
        new_path = os.path.join(parent_dir, new_name)
        os.rename(old_path, new_path)
