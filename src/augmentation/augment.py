import os
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Augmentation pipeline (applied then SAVED)
aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),

    transforms.RandomRotation(5),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.02, 0.02),
        scale=(0.95, 1.05),
        shear=2
    ),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def augment_folder(input_dir: str, output_dir: str, k: int = 5):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Walk through all images recursively
    for img_path in input_dir.rglob("*"):
        if not img_path.is_file() or not is_image_file(img_path):
            continue

        # Keep same relative folder structure
        rel_path = img_path.relative_to(input_dir)
        out_subdir = output_dir / rel_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        # Load image
        img = Image.open(img_path).convert("RGB")

        stem = img_path.stem
        suffix = img_path.suffix

        # Save original too (optional)
        # img.save(out_subdir / f"{stem}_orig{suffix}")

        # Create k augmented versions
        for i in range(k):
            aug_img = aug(img)
            out_path = out_subdir / f"{stem}_aug{i+1}{suffix}"
            aug_img.save(out_path)

    print(f"Done. Saved augmented images to: {output_dir}")

if __name__ == "__main__":
    # Example (change these paths)
    input_dir = r"C:\Users\diya.abraham\project dataset\Signatures\Sreya\Genuine"           # e.g., data/train/person1/genuine, forged...
    output_dir = r"C:\Users\diya.abraham\project dataset\Signatures\Sreya\Genuine-Augmented"
    k = 5  # how many new images per original

    augment_folder(input_dir, output_dir, k=k)