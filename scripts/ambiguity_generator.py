import os
import numpy as np
import cv2
from skimage import draw
from tqdm import tqdm
import random

np.random.seed(42)
random.seed(42)

IMG_SIZE = 224
SAMPLES_PER_CLASS = 1200

BASE_PATH = "../data/raw_dataset"

classes = {
    0: "no_corrosion",
    1: "mild",
    2: "moderate",
    3: "severe"
}


def apply_environmental_noise(img):
    # Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 10, img.shape)
        img = img + noise

    # Salt & pepper noise
    if random.random() < 0.3:
        salt_pepper = np.random.rand(*img.shape[:2])
        img[salt_pepper < 0.01] = 0
        img[salt_pepper > 0.99] = 255

    # Illumination scaling
    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        img = img * factor

    # Blur
    if random.random() < 0.4:
        img = cv2.GaussianBlur(img.astype(np.uint8), (3, 3), 0)

    return np.clip(img, 0, 255).astype(np.uint8)


def create_base_surface():
    base = np.random.normal(180, 15, (IMG_SIZE, IMG_SIZE, 3))
    gradient = np.tile(np.linspace(0.7, 1.3, IMG_SIZE), (IMG_SIZE, 1))
    gradient = np.expand_dims(gradient, axis=2)
    base = base * gradient
    return np.clip(base, 0, 255).astype(np.uint8)


def add_corrosion_patch(img, intensity):
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    num_seeds = intensity * random.randint(6, 12)

    for _ in range(num_seeds):
        r = random.randint(5, 20 + intensity * 5)
        x = random.randint(r, IMG_SIZE - r)
        y = random.randint(r, IMG_SIZE - r)
        rr, cc = draw.disk((x, y), r)
        mask[rr, cc] = 255

    kernel = np.ones((3 + intensity * 2, 3 + intensity * 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=intensity)

    rust_texture = np.random.normal(110, 30, (IMG_SIZE, IMG_SIZE))
    rust_texture = np.clip(rust_texture, 50, 160)

    for c in range(3):
        img[:, :, c] = np.where(mask == 255, rust_texture, img[:, :, c])

    return img


def generate():
    for label, name in classes.items():
        folder = os.path.join(BASE_PATH, name)
        os.makedirs(folder, exist_ok=True)

        for i in tqdm(range(SAMPLES_PER_CLASS), desc=name):
            base = create_base_surface()

            # Add primary corrosion
            if label > 0:
                base = add_corrosion_patch(base, label)

            # Add ambiguity (severity mixing)
            if label == 1 and random.random() < 0.15:
                base = add_corrosion_patch(base, 2)

            if label == 2 and random.random() < 0.2:
                base = add_corrosion_patch(base, 3)

            if label == 3 and random.random() < 0.1:
                base = add_corrosion_patch(base, 2)

            base = apply_environmental_noise(base)

            cv2.imwrite(os.path.join(folder, f"{name}_{i}.png"), base)


if __name__ == "__main__":
    generate()
