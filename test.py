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
    import re
    import matplotlib.pyplot as plt

    # Đường dẫn tới tệp log của bạn
    log_file_path = 'training.log.txt'

    # Đọc nội dung tệp log
    with open(log_file_path, 'r') as file:
        log_data = file.readlines()

    # Khởi tạo danh sách để lưu trữ LearningRate và Global Step
    learning_rates = []
    global_steps = []

    # Duyệt qua từng dòng trong log và tách giá trị LearningRate và Global Step
    for line in log_data:
        if 'LearningRate' in line and 'Global Step' in line:
            lr_match = re.search(r'Loss (\d+\.\d+)', line)
            step_match = re.search(r'Epoch: (\d+)', line)
            if lr_match and step_match:
                learning_rate = float(lr_match.group(1))
                global_step = int(step_match.group(1))
                learning_rates.append(learning_rate)
                global_steps.append(global_step)

    # Vẽ đồ thị bằng Matplotlib
    fontsize = 15
    plt.figure(figsize=(6, 6))
    plt.plot(global_steps, learning_rates, linestyle='-', linewidth=2)
    plt.xlabel('Iterations', fontsize=fontsize)  # x_label
    plt.ylabel("Learning Rate", fontsize=fontsize)  # y_label
    plt.title("Learning Rate vs Iterations", fontsize=fontsize)
    plt.grid(True)
    plt.savefig("learning_rate_plot.png", dpi=600, bbox_inches='tight')
    plt.show()