from os import getcwd, path

TOKENIZER_OUTPUT_PATH = path.join(getcwd(), "out")
VOCAB_INPUT_PATH = path.join(TOKENIZER_OUTPUT_PATH, "vovab.json")
MERGES_INPUT_PATH = path.join(TOKENIZER_OUTPUT_PATH, "merges.txt")
