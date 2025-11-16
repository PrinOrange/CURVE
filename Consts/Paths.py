from os import getcwd, path

CWD = getcwd()

# Evaluation output
REPORT_DIR = path.join(CWD, "report")

# Model training output
MODELS_DIR = path.join(CWD, "out")

# Pretokenization output
VOCAB_FILE_PATH = path.join(MODELS_DIR, "vovab.json")
MERGES_FILE_PATH = path.join(MODELS_DIR, "merges.txt")
