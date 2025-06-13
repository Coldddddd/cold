import os
from PIL import Image
from torch.utils import data

class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path: str, transform=None):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        self.imgs_path = [line.split()[0] for line in lines]
        self.labels = [int(line.split()[1].strip()) for line in lines]
        self.transform = transform
        self.data_dir = os.path.dirname(txt_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path, label = self.imgs_path[idx], self.labels[idx]
        img_path = os.path.join(self.data_dir, img_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
