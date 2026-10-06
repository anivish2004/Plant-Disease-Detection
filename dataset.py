"""Discover and split PlantVillage image paths without loading pixels."""
import random
from pathlib import Path

DEFAULT_DATASET = Path.home() / "Desktop/PlantVillage-Dataset/raw/color"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_classes(dataset_path=DEFAULT_DATASET):
    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    classes = {}
    for directory in sorted(root.iterdir()):
        if directory.is_dir():
            images = sorted(str(p) for p in directory.iterdir()
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if not images:
                raise ValueError(f"Class has no supported images: {directory.name}")
            classes[directory.name] = images
    if len(classes) < 2:
        raise ValueError("At least two nonempty class folders are required.")
    return classes


def split_dataset(classes, seed=42):
    """Stratify each class into 70% training, 15% validation, 15% test."""
    rng = random.Random(seed)
    splits = {name: ([], []) for name in ("train", "validation", "test")}
    for label, (name, image_paths) in enumerate(classes.items()):
        paths = list(image_paths)
        if len(paths) < 3:
            raise ValueError(f"Class needs at least three images: {name}")
        rng.shuffle(paths)
        held_out = max(1, int(len(paths) * 0.15))
        subsets = {"test": paths[:held_out], "validation": paths[held_out:2 * held_out],
                   "train": paths[2 * held_out:]}
        for split, subset in subsets.items():
            splits[split][0].extend(subset)
            splits[split][1].extend([label] * len(subset))
    return splits
