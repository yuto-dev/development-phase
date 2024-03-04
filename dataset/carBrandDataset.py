from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class CarBrandDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.images.append((os.path.join(class_dir, img_name), class_name))
        
        # Debugging: Print the classes and images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_index = self.classes.index(label)
        return image, label_index
