import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class SequenceDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.root_dir = root_dir
        self.sequence_folders = os.listdir(root_dir)

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        self.class_to_idx = {
            "no_corrosion": 0,
            "mild": 1,
            "moderate": 2,
            "severe": 3
        }

    def __len__(self):
        return len(self.sequence_folders)

    def __getitem__(self, idx):
        seq_folder = self.sequence_folders[idx]
        seq_path = os.path.join(self.root_dir, seq_folder)

        frames = []
        for i in range(5):
            img_path = os.path.join(seq_path, f"frame_{i}.png")
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            frames.append(image)

        frames = torch.stack(frames)

        with open(os.path.join(seq_path, "label.txt"), "r") as f:
            label_name = f.read().strip()

        label = self.class_to_idx[label_name]

        return frames, label
