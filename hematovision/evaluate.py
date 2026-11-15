"""Model evaluation: accuracy, classification report, confusion matrix."""
import argparse
import json

import numpy as np

from .config import CLASS_NAMES_PATH, DATA_DIR, MODEL_PATH, REPORT_DIR


def evaluate_model(model, test_ds, class_names, out_dir=REPORT_DIR) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    out_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    labels_idx = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels_idx, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    print(classification_report(
        y_true, y_pred, labels=labels_idx, target_names=class_names, zero_division=0,
    ))
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels_idx, class_names, rotation=45, ha="right")
    ax.set_yticks(labels_idx, class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in labels_idx:
        for j in labels_idx:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return report


def main():
    import tensorflow as tf
    from .data import load_datasets

    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_NAMES_PATH.read_text())
    _, _, test_ds, _ = load_datasets(args.data_dir)
    evaluate_model(model, test_ds, class_names)
    print(f"Reports saved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
