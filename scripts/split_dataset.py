import os
import random
import shutil
from tqdm import tqdm

RAW_PATH = "../data/raw_dataset"
PROCESSED_PATH = "../data/processed"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)


def split_class(class_name):
    class_path = os.path.join(RAW_PATH, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    split_map = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in split_map.items():
        target_folder = os.path.join(PROCESSED_PATH, split_name, class_name)
        os.makedirs(target_folder, exist_ok=True)

        for img in tqdm(split_images, desc=f"{class_name}-{split_name}"):
            src = os.path.join(class_path, img)
            dst = os.path.join(target_folder, img)
            shutil.copy(src, dst)


def main():
    class_names = os.listdir(RAW_PATH)
    for class_name in class_names:
        split_class(class_name)


if __name__ == "__main__":
    main()
