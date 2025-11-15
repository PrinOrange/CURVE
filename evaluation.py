import pandas as pd
import numpy as np
import csv
import pickle
import re
import torch
import sklearn
import os
import random
import clang
from clang import *
from clang import cindex
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM, RobertaForSequenceClassification
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset, pretoke
from transformers.modeling_outputs import SequenceClassifierOutput
from tokenizers import Tokenizer
from tokenizers import normalizers, decoders
from tokenizers.normalizers import StripAccents, unicode_normalizer_from_str, Replace
from tokenizers.processors import TemplateProcessing
from tokenizers import processors, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import NormalizedString, PreTokenizedString
from typing import List
from Consts.Paths import VOCAB_INPUT_PATH, MERGES_INPUT_PATH
from Consts.Tokens import ROBERTA_SPECIAL_TOKENS
from CweBert.Tokenizer import CweBertTokenizer
from CweBert.DataCollator import CweBERTCollatorForLanguageModeling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

vocab, merges = BPE.read_file(vocab=VOCAB_INPUT_PATH, merges=MERGES_INPUT_PATH)
my_tokenizer = Tokenizer(BPE(vocab, merges))
my_tokenizer.normalizer = normalizers.Sequence([StripAccents(), Replace(" ", "Ä")])
my_tokenizer.pre_tokenizer = PreTokenizer.custom(CweBertTokenizer())
my_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
my_tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[
        (token, index) for index, token in enumerate(ROBERTA_SPECIAL_TOKENS)
    ],
)

model = RobertaForSequenceClassification.from_pretrained("./models/VB-MLP_%s")
print(model.num_parameters())
