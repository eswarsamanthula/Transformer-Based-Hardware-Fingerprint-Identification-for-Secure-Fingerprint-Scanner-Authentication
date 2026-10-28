"""
config.py — single source of truth for every hyperparameter.

Every value here mirrors what "Robust Authentication: Leveraging Hardware
Fingerprints and AI to Enhance Security Against Spoofing" (the base paper,
IEEE Access 2025) actually did. The only thing this project changes is
WHICH models get trained (transformer backbones instead of / in addition
to the paper's CNNs). Nothing about the data, the split, the loss, the
optimizer, or the lack of augmentation is different.

Paper facts this file encodes (see paper Section VI-B):
  - Raw images, no data augmentation. (Resizing to each model's required
    input size and [0,1] tensor conversion still happen in dataset.py —
    that's a format requirement, not augmentation. See dataset.py's
    docstring for the precise wording.)
  - Trained from scratch (pretrained=False for every model below) — the
    paper trains every model, CNN or otherwise, from random weights.
  - ONE exception, the project's own follow-up: EfficientFormer-L1 (RGB) —
    its official run uses ImageNet-pretrained weights (pretrained=True,
    AdamW lr=1e-4, 20 epochs, best at epoch 12). This is a reported
    deviation from the paper protocol; see the model's registry entry.
  - Adam optimizer, categorical cross-entropy (multiclass) / binary
    cross-entropy (binary), categorical accuracy as the tracked metric.
  - Early stopping on validation loss, patience = 10, best weights restored.
  - 70/30 train/validation split on Dataset 1 (per-scanner stratified).
  - Dataset 2 (~200 images, unseen users, files renamed
    Sensor<n>(Test1)<k>=User_A, Sensor<n>(Test2)<k>=User_B)
    used ONLY as the final unseen-user test set — never train or val.
  - Two channel modes trained and reported separately: grayscale (1ch)
    and RGB (3ch).
  - Batch size 8 or 32 depending on the model (paper used 8 or 32; we
    default to 32 for the lighter backbones and 8 for the largest one,
    matching the paper's own reasoning: "depending on the computational
    requirements of each model").
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Dataset 1 — the paper's R307 corpus: 2,019 BMP images in 10 scanner
# folders. The folder name IS the class label (10-class scanner ID). These
# files carry no user id in their names (fingerprint0.bmp ...), so no
# per-user assignment is attempted anywhere — Dataset 1 is train/val only.
DATA_ROOT = os.environ.get("THFSA_DATA_ROOT", "../dataset/Dataset1/Fingerprint")
# Expected layout, exactly one folder per scanner:
#   DATA_ROOT/Sensor1/*.bmp
#   ...
#   DATA_ROOT/Sensor10/*.bmp

# Dataset 2 — the independent unseen-user test set: 200 BMP images named
# Sensor<n>(Test<m>)<k>.bmp — (Test1) = User A, (Test2) = User B
# (10 captures × 2 users × 10 sensors). Used ONLY as the final test set;
# never for train or val.
TEST_DATA_ROOT = os.environ.get(
    "THFSA_TEST_DATA_ROOT",
    "../dataset/Dataset2/Fingerprint_of_Test1_and_Test2",
)

CHECKPOINT_DIR = "../checkpoints"          # CHECKPOINT_DIR/cnn/... and CHECKPOINT_DIR/transformer/...
LOG_DIR = "../logs"                         # LOG_DIR/training/... (per-epoch CSVs)
RESULTS_DIR = "../results"                  # RESULTS_DIR/metrics, /graphs, /tables

NUM_SCANNERS = 10
IMG_ORIG_SIZE = (288, 256)  # (H, W) as captured — paper's raw BMP resolution

# ---------------------------------------------------------------------------
# Train/val split (on the remaining data, after holdout is removed)
# ---------------------------------------------------------------------------
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.30
SPLIT_SEED = 42  # fixed so the split is reproducible across runs

# ---------------------------------------------------------------------------
# Channel modes to train and report separately
# ---------------------------------------------------------------------------
CHANNEL_MODES = ["grayscale", "rgb"]  # 1ch and 3ch, run + reported independently

# ---------------------------------------------------------------------------
# Optimizer / loss / training loop — identical across every model
# ---------------------------------------------------------------------------
OPTIMIZER = "adam"
LEARNING_RATE = 0.001       # DEFAULT fallback only — per-model lr lives in
                            # MODEL_REGISTRY[model]["lr"], which train.py reads
                            # (VGG16 and LeNet5 use 0.0001; the paper states it)
EARLY_STOP_PATIENCE = 10       # epochs, monitored on val loss
MAX_EPOCHS = 100                # DEFAULT fallback only — per-model cap lives
                                # in MODEL_REGISTRY[model]["max_epochs"], which
                                # train.py reads (efficientnet 51, vgg16 54,
                                # lenet5 30, mobilenet/resnet50 40, resnet152 60,
                                # transformers 50-70). Early stopping will cut
                                # runs short in practice.
USE_AUGMENTATION = False        # paper: "raw dataset without any data augmentation"

# ---------------------------------------------------------------------------
# Model registry — this is the ONLY section that differs from the base paper.
# The paper's 6 CNNs are kept as baselines; all 8 transformer / hybrid-CNN
# backbones (both prior candidate sets) are this project's addition
# (per the Phase II proposal, Objective 1) and are all kept — none removed.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # --- base paper's CNN baselines (Objective 2 comparison) ---
    # pretrained=False on every model: paper explicitly trains from scratch.
    # lr per paper Section VI: 0.001 everywhere except VGG16 and LeNet5 (0.0001).
    # max_epochs per paper Section VI (epoch counts the paper actually ran);
    # MAX_EPOCHS is the fallback for models the paper didn't specify.
    "efficientnet_b3": {"batch_size": 8,  "timm_name": "efficientnet_b3",              "pretrained": False, "lr": 0.001,  "max_epochs": 51},
    "mobilenet":       {"batch_size": 16, "timm_name": "mobilenetv2_100",              "pretrained": False, "lr": 0.001,  "max_epochs": 40},
    "resnet50":        {"batch_size": 16, "timm_name": "resnet50",                     "pretrained": False, "lr": 0.001,  "max_epochs": 40},
    "resnet152":       {"batch_size": 16, "timm_name": "resnet152",                    "pretrained": False, "lr": 0.001,  "max_epochs": 60},
    "vgg16":           {"batch_size": 8,  "timm_name": "vgg16",                        "pretrained": False, "lr": 0.0001, "max_epochs": 54},  # paper: lr 0.0001
    "lenet5":          {"batch_size": 32, "timm_name": None,                           "pretrained": False, "lr": 0.0001, "max_epochs": 30},  # custom, see models.py; paper: lr 0.0001

    # --- this project's transformer / hybrid-CNN backbones (Objective 1) ---
    # lr 0.001 to match the paper's default across all trained-from-scratch models
    "vit_b16":         {"batch_size": 8,  "timm_name": "vit_base_patch16_224",         "pretrained": False, "lr": 0.001,  "max_epochs": 70},
    "swin_tiny":       {"batch_size": 32, "timm_name": "swin_tiny_patch4_window7_224", "pretrained": False, "lr": 0.001,  "max_epochs": 50},
    "deit_small":      {"batch_size": 32, "timm_name": "deit_small_patch16_224",       "pretrained": False, "lr": 0.001,  "max_epochs": 50},
    "convnext_tiny":   {"batch_size": 32, "timm_name": "convnext_tiny",                "pretrained": False, "lr": 0.001,  "max_epochs": 50},

    # --- lightweight transformer / hybrid-Transformer backbones (kept alongside the above) ---
    # Patience is global (10); these per-model values are maximum epoch ceilings.
    "tinyvit_5m":         {"batch_size": 32, "timm_name": "tiny_vit_5m_224.in1k",                    "pretrained": False, "lr": 0.001, "max_epochs": 60},
    "mobilevit_xs":       {"batch_size": 32, "timm_name": "mobilevit_xs.cvnets_in1k",                "pretrained": False, "lr": 0.001, "max_epochs": 60},
    "efficientformer_l1": {"batch_size": 32, "timm_name": "efficientformer_l1.snap_dist_in1k",       "pretrained": True,  "lr": 0.0001, "max_epochs": 20},  # official run: ImageNet-pretrained, AdamW lr=1e-4, best epoch 12
    "pit_ti":             {"batch_size": 32, "timm_name": "pit_ti_224.in1k",                         "pretrained": False, "lr": 0.001, "max_epochs": 60},
}

# Standard input size most timm models expect. Images are resized to this —
# resizing is a format requirement, not augmentation (no crop/flip/jitter).
MODEL_INPUT_SIZE = {
    "efficientnet_b3": 300,
    "mobilenet": 224,
    "resnet50": 224,
    "resnet152": 224,
    "vgg16": 224,
    "lenet5": 32,
    "vit_b16": 224,
    "swin_tiny": 224,
    "deit_small": 224,
    "convnext_tiny": 224,
    "tinyvit_5m": 224,
    "mobilevit_xs": 256,
    "efficientformer_l1": 224,
    "pit_ti": 224,
}

# Explicit group lists — used to route checkpoints into checkpoints/cnn/ vs
# checkpoints/transformer/, and for anything that needs to treat the two
# groups separately (e.g. the Objective 2 comparison table).
CNN_MODELS = ["efficientnet_b3", "mobilenet", "resnet50", "resnet152", "vgg16", "lenet5"]
TRANSFORMER_MODELS = [
    "vit_b16", "swin_tiny", "deit_small", "convnext_tiny",
    "tinyvit_5m", "mobilevit_xs", "efficientformer_l1", "pit_ti",
]


def model_category(model_name: str) -> str:
    if model_name in CNN_MODELS:
        return "cnn"
    if model_name in TRANSFORMER_MODELS:
        return "transformer"
    raise ValueError(f"'{model_name}' is not in CNN_MODELS or TRANSFORMER_MODELS")
