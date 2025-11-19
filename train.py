import torch


RANDOM_SEED = 42
EPOCHS = 3
BATCH_SIZE = 64

CHECK_STEPS = 2000
LOGGING_STEPS = 500
EARLY_STOP_PATIENT = 5

HF_DATASET_PATH = "codemetic/curve"
HF_DATASET_SUBSET = "draper"
HF_MODEL_PATH = "codemetic/cwebert-pretrained"

MAX_TOKEN_LENGTH = 512

OUTPUT_DIR = f"./out/{HF_DATASET_SUBSET}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Fine tuneing CWEBERT with the CURVE Framework on dataset: {HF_DATASET_SUBSET}")
print("Using device:", DEVICE)
