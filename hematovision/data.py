"""Dataset loading and preprocessing.

Supported layouts under ``data/``:

1. Kaggle "Blood Cell Images" style::

       data/train/<class>/*.jpeg
       data/test/<class>/*.jpeg      (optional)

2. A single folder of class sub-folders::

       data/<class>/*.jpg

   In this case the data is split ~70/15/15 into train/val/test.
"""
from pathlib import Path

from .config import BATCH_SIZE, IMG_SIZE, SEED


def _has_classes(path: Path) -> bool:
    return path.is_dir() and any(p.is_dir() for p in path.iterdir())


def load_datasets(data_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, seed=SEED):
    """Return (train_ds, val_ds, test_ds, class_names)."""
    import tensorflow as tf

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    common = dict(
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        seed=seed,
    )
    train_root = data_dir / "train"
    test_root = data_dir / "test"

    if _has_classes(train_root):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_root, validation_split=0.15, subset="training", **common
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_root, validation_split=0.15, subset="validation", **common
        )
        if _has_classes(test_root):
            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_root, shuffle=False, class_names=train_ds.class_names,
                **{k: v for k, v in common.items() if k != "seed"},
            )
        else:
            test_ds = val_ds
    elif _has_classes(data_dir):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir, validation_split=0.3, subset="training", **common
        )
        holdout = tf.keras.utils.image_dataset_from_directory(
            data_dir, validation_split=0.3, subset="validation", **common
        )
        n = int(tf.data.experimental.cardinality(holdout))
        half = max(n // 2, 1)
        val_ds = holdout.take(half)
        test_ds = holdout.skip(half) if n > 1 else holdout
    else:
        raise FileNotFoundError(
            f"No class folders found in {data_dir}. See README for the expected layout."
        )

    class_names = list(train_ds.class_names)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names
