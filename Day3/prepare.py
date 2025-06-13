import os

# 文件路径写入功能
def generate_file_list(base_dir, output_file):
    with open(output_file, 'w') as file:
        # 遍历文件夹及其子目录
        for index, folder in enumerate(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path):
                # 获取并写入图片路径与标签
                image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
                for img in image_paths:
                    file.write(f"{img} {index}\n")

# 创建训练集和验证集文件
generate_file_list(r'/image2/train', 'train.txt')
generate_file_list(r'/image2/val', 'val.txt')
