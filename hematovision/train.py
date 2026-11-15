"""Train the CNN.   Usage:  python -m hematovision.train --data-dir data --epochs 25"""
import argparse
import json

from .config import (
    BATCH_SIZE, CLASS_NAMES_PATH, DATA_DIR, EPOCHS, MODEL_DIR, MODEL_PATH, REPORT_DIR, SEED,
)


def plot_history(history, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("accuracy", "loss")):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history[f"val_{key}"], label="validation")
        ax.set_title(key.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_history.png", dpi=150)
    plt.close(fig)


def main():
    import tensorflow as tf
    from .data import load_datasets
    from .evaluate import evaluate_model
    from .model import build_model

    parser = argparse.ArgumentParser(description="Train HematoVision CNN.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_datasets(args.data_dir, args.batch_size)
    print(f"Classes: {class_names}")
    CLASS_NAMES_PATH.write_text(json.dumps(class_names))

    model = build_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    plot_history(history, REPORT_DIR)

    model.save(MODEL_PATH)  # ensure the final (best-restored) weights are saved
    print(f"Model saved to {MODEL_PATH}")

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {loss:.4f}  Test accuracy: {acc:.4f}")
    evaluate_model(model, test_ds, class_names)


if __name__ == "__main__":
    main()
