import os
import numpy as np
import cv2
from skimage import draw
from tqdm import tqdm

BASE_PATH = "../data/processed"
IMG_SIZE = 224
SAMPLES_PER_CLASS = 500

classes = {
    0: "no_corrosion",
    1: "mild",
    2: "moderate",
    3: "severe"
}

def create_base_metal():
    base = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 200
    noise = np.random.randint(0, 20, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return cv2.add(base, noise)

def add_corrosion(image, severity):
    img = image.copy()
    num_spots = severity * 10
    for _ in range(num_spots):
        r = np.random.randint(5, 20 + severity * 5)
        x = np.random.randint(r, IMG_SIZE - r)
        y = np.random.randint(r, IMG_SIZE - r)
        rr, cc = draw.disk((x, y), r)
        img[rr, cc] = [139, 69, 19]  # Rust color
    return img

def generate():
    for label, name in classes.items():
        folder = os.path.join(BASE_PATH, "train", name)
        os.makedirs(folder, exist_ok=True)

        for i in tqdm(range(SAMPLES_PER_CLASS), desc=name):
            base = create_base_metal()
            if label > 0:
                img = add_corrosion(base, label)
            else:
                img = base
            cv2.imwrite(os.path.join(folder, f"{name}_{i}.png"), img)

if __name__ == "__main__":
    generate()
