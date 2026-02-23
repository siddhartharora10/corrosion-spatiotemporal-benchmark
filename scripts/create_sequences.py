import os
import shutil
import random

BASE_PATH = "../data/processed"
SEQ_PATH = "../data/sequences"
SEQUENCE_LENGTH = 5

random.seed(42)


def create_sequences(split):
    split_path = os.path.join(BASE_PATH, split)
    output_split_path = os.path.join(SEQ_PATH, split)
    os.makedirs(output_split_path, exist_ok=True)

    sequence_id = 0

    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        images = sorted(os.listdir(class_path))

        # Group into sequences
        for i in range(0, len(images) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
            sequence_folder = os.path.join(output_split_path, f"seq_{sequence_id}")
            os.makedirs(sequence_folder, exist_ok=True)

            for j in range(SEQUENCE_LENGTH):
                src = os.path.join(class_path, images[i + j])
                dst = os.path.join(sequence_folder, f"frame_{j}.png")
                shutil.copy(src, dst)

            # Save label file
            with open(os.path.join(sequence_folder, "label.txt"), "w") as f:
                f.write(class_name)

            sequence_id += 1


def main():
    for split in ["train", "val", "test"]:
        create_sequences(split)


if __name__ == "__main__":
    main()
