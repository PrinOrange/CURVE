from os import environ

from Consts.Paths import MODELS_DIR
from Consts.Tokens import ROBERTA_SPECIAL_TOKENS, SPECIAL_TOKENS
from CweBert.Tokenizer import CweBertTokenizer
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

# Enable multi-thread BPE procession
environ["TOKENIZERS_PARALLELISM"] = "true"

# Huggingface dataset configuration
HF_DATASET_PATH = "codemetic/curve"
HF_DATASET_SUBSET = "pretrain"
HF_DATASET_SPILT = "train"

# Program default args
DEFAULT_PROGRAM_ARGS = {
    # The size of the vocabularies. For code corpus, the size 30,000 is enough.
    "vocab_size": 50_000,
    # The minimum frequency of the token that should be attended to.
    "min_frequency": 3,
    # Enable this option if you want to load dataset in streaming rather than download full dataset.
    "streaming_load": False,
}


def start_pretokenization():
    print("Start Pretraining tokenizer and generating vocabulary.")
    print("It may take some times...")

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = PreTokenizer.custom(CweBertTokenizer())
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[(tok, i) for i, tok in enumerate(ROBERTA_SPECIAL_TOKENS)],
    )

    code_corpus = load_dataset(
        path=HF_DATASET_PATH,
        name=HF_DATASET_SUBSET,
        split=HF_DATASET_SPILT,
        streaming=DEFAULT_PROGRAM_ARGS["streaming_load"],
    )["source"]

    trainer = BpeTrainer(
        vocab_size=DEFAULT_PROGRAM_ARGS["vocab_size"],
        min_frequency=DEFAULT_PROGRAM_ARGS["min_frequency"],
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(code_corpus, trainer)
    tokenizer.model.save(MODELS_DIR)


if __name__ == "__main__":
    start_pretokenization()
