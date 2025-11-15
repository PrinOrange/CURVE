import csv
import os
import pickle
import random
import re
from pathlib import Path
from typing import List

import clang
import numpy as np
import pandas as pd
import sklearn
import torch
from clang import *
from clang import cindex
from Consts.Paths import MERGES_FILE_PATH, VOCAB_FILE_PATH
from Consts.Tokens import ROBERTA_SPECIAL_TOKENS
from CweBert.DataCollator import CweBERTCollatorForLanguageModeling
from CweBert.Tokenizer import CweBertTokenizer
from tokenizers import (ByteLevelBPETokenizer, NormalizedString,
                        PreTokenizedString, Tokenizer, decoders, normalizers,
                        pre_tokenizers, processors)
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import (Replace, StripAccents,
                                    unicode_normalizer_from_str)
from tokenizers.pre_tokenizers import PreTokenizer, Whitespace
from tokenizers.processors import BertProcessing, TemplateProcessing
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import (DataCollatorForLanguageModeling,
                          LineByLineTextDataset, RobertaConfig,
                          RobertaForMaskedLM, RobertaForSequenceClassification,
                          RobertaTokenizerFast, Trainer, TrainingArguments,
                          pretoke)
from transformers.modeling_outputs import SequenceClassifierOutput

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

vocab, merges = BPE.read_file(vocab=VOCAB_FILE_PATH, merges=MERGES_FILE_PATH)

tokenizer = Tokenizer(BPE(vocab, merges))
tokenizer.pre_tokenizer = PreTokenizer.custom(CweBertTokenizer())
tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[(tok, i) for i, tok in enumerate(ROBERTA_SPECIAL_TOKENS)],
)

model = RobertaForSequenceClassification.from_pretrained("./models/VB-MLP_%s")
print(model.num_parameters())
