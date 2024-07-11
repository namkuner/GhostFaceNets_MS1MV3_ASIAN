import os

def create_list_file(data_dir, list_file):
    with open(list_file, 'w') as f:
        idx = 0
        for label, class_name in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_name, image_name)
                    f.write(f"{idx}\t{label}\t{image_path}\n")
                    idx += 1

# Tạo file train.lst
create_list_file("test_dataset", "small_dataset/train.lst")