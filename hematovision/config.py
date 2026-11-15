"""Central configuration for HematoVision."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
MODEL_PATH = MODEL_DIR / "hematovision.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# Model / training
IMG_SIZE = (128, 128)          # (height, width)
BATCH_SIZE = 32
EPOCHS = 25
SEED = 42

# Default class names (alphabetical, same as folder order in the Kaggle dataset)
DEFAULT_CLASSES = ["eosinophil", "lymphocyte", "monocyte", "neutrophil"]

# Upload validation
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024   # 5 MB
MIN_DIMENSION = 32                    # px
MAX_DIMENSION = 8000                  # px
MIN_PIXEL_STD = 3.0                   # rejects blank / uniform images

# Prediction
CONFIDENCE_THRESHOLD = 0.60           # below this the result is flagged "uncertain"
