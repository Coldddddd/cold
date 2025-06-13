import os
import shutil
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random.seed(42)

# Paths
dataset_dir = r'/image2'
train_dir = r'/image2/train'
val_dir = r'/image2/val'

# Split ratio
train_ratio = 0.7

# Create directories for train and val sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Process each class
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_path) and class_name not in ["train", "val"]:
        
        # Get all image paths
        images = [os.path.join(class_name, f) for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Split into train and val
        train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)

        # Create subdirectories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Move images to appropriate directories
        for img, dst_dir in zip([train_images, val_images], [train_dir, val_dir]):
            for img_path in img:
                shutil.move(os.path.join(dataset_dir, img_path), os.path.join(dst_dir, img_path))

        # Clean up class directory
        shutil.rmtree(class_path)
