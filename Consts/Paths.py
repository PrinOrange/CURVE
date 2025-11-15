from os import getcwd, path

MODELS_DIR = path.join(getcwd(), "out")
VOCAB_FILE_PATH = path.join(MODELS_DIR, "vovab.json")
MERGES_FILE_PATH = path.join(MODELS_DIR, "merges.txt")
