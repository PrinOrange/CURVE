from clang import *
from clang import cindex
from tokenizers import NormalizedString, PreTokenizedString
from typing import List


class CweBertTokenizer:
    cidx = cindex.Index.create()

    def clang_split(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        tokens = []
        translationUnit = self.cidx.parse(
            "tmp.c",
            args=[""],
            unsaved_files=[("tmp.c", str(normalized_string.original))],
            options=0,
        )
        for t in translationUnit.get_tokens(extent=translationUnit.cursor.extent):
            spelling = t.spelling.strip()
            if spelling == "":
                continue
            tokens.append(NormalizedString(spelling))
        return tokens

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.clang_split)
