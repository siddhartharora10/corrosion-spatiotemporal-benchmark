import os
import numpy as np
import cv2
from skimage import draw
from tqdm import tqdm

IMG_SIZE = 224
SAMPLES_PER_CLASS = 1000

BASE_PATH = "../data/raw_dataset"

classes = {
    0: "no_corrosion",
    1: "mild",
    2: "moderate",
    3: "severe"
}


def add_lighting_gradient(img):
    gradient = np.tile(np.linspace(0.7, 1.2, IMG_SIZE), (IMG_SIZE, 1))
    gradient = np.expand_dims(gradient, axis=2)
    img = img * gradient
    return np.clip(img, 0, 255).astype(np.uint8)


def create_textured_metal():
    base = np.random.normal(180, 10, (IMG_SIZE, IMG_SIZE, 3))
    base = add_lighting_gradient(base)
    return base.astype(np.uint8)


def add_corrosion_spread(image, severity):
    img = image.copy()

    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    num_seeds = severity * 8

    for _ in range(num_seeds):
        r = np.random.randint(5, 15 + severity * 5)
        x = np.random.randint(r, IMG_SIZE - r)
        y = np.random.randint(r, IMG_SIZE - r)
        rr, cc = draw.disk((x, y), r)
        mask[rr, cc] = 255

    kernel_size = 3 + severity * 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=severity)

    rust_texture = np.random.normal(120, 25, (IMG_SIZE, IMG_SIZE))
    rust_texture = np.clip(rust_texture, 60, 160)

    for c in range(3):
        img[:, :, c] = np.where(
            mask == 255,
            rust_texture,
            img[:, :, c]
        )

    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img.astype(np.uint8)


def generate():
    for label, name in classes.items():
        folder = os.path.join(BASE_PATH, name)
        os.makedirs(folder, exist_ok=True)

        for i in tqdm(range(SAMPLES_PER_CLASS), desc=name):
            base = create_textured_metal()
            if label > 0:
                img = add_corrosion_spread(base, label)
            else:
                img = base

            cv2.imwrite(os.path.join(folder, f"{name}_{i}.png"), img)


if __name__ == "__main__":
    generate()
