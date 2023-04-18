import os
import shutil
from typing import Iterable, List, Optional, Tuple

import datasets
from datasets import Dataset, DatasetDict, Sequence, Value
from tango import Step
from tango.common import Lazy
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.transformers import Tokenizer

from loreiba.tokenizers import train_tokenizer


@Step.register("loreiba.data.tokenize::tokenize_plus")
class TokenizePlus(Step):
    """
    Use a pretrained transformer tokenizer to get inputs necessary for the language model. Also,
    note which wordpieces belong to whole tokens in the original tokenization.
    """

    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def _intra_word_tokenize(
        self,
        string_tokens: List[str],
        tokenizer: Tokenizer,
        max_wordpieces: int,
    ) -> Tuple[List[int], List[Optional[Tuple[int, int]]]]:
        tokens = []
        offsets = []
        for token_string in string_tokens:
            wordpieces = tokenizer.encode_plus(
                token_string,
                add_special_tokens=False,
                return_tensors=None,
                return_offsets_mapping=False,
                return_attention_mask=False,
            )
            wp_ids = wordpieces["input_ids"]

            # Stop early if adding this token would exceed our budget
            if len(tokens) + len(wp_ids) > max_wordpieces:
                break

            if len(wp_ids) > 0:
                tokens.extend(wp_ids)
                offsets.append((len(tokens) - len(wp_ids), len(tokens) - 1))
            else:
                tokens.append(tokenizer.unk_token_id)
                offsets.append((len(tokens) - 1, len(tokens) - 1))
        return tokens, offsets

    @staticmethod
    def _increment_offsets(
        offsets: Iterable[Optional[Tuple[int, int]]], increment: int
    ) -> List[Optional[Tuple[int, int]]]:
        return [None if offset is None else (offset[0] + increment, offset[1] + increment) for offset in offsets]

    def intra_word_tokenize(
        self,
        string_tokens: List[str],
        tokenizer: Tokenizer,
        max_wordpieces: int,
    ) -> Tuple[List[int], List[Optional[Tuple[int, int]]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that tokens[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.
        This function inserts special tokens.
        """
        wp_ids, offsets = self._intra_word_tokenize(string_tokens, tokenizer, max_wordpieces - 2)
        # Handle special tokens
        wp_ids = [tokenizer.cls_token_id] + wp_ids + [tokenizer.sep_token_id]
        offsets = self._increment_offsets(offsets, 1)
        offsets = [(0, 0)] + offsets + [(offsets[-1][1] + 1,) * 2]
        return wp_ids, offsets

    def _process_split(
        self, split: Dataset, tokenizer: Tokenizer, max_length: Optional[int], token_column: str
    ) -> Dataset:
        def inner():
            for d in split:
                sentence = d[token_column]
                wp_ids, token_spans = self.intra_word_tokenize(sentence, tokenizer, max_length)
                flattened = []
                for pair in token_spans:
                    flattened.extend(pair)
                d = {
                    **d,
                    "input_ids": wp_ids,
                    "token_spans": flattened,
                    token_column: sentence[: len(token_spans) - 2],
                    "attention_mask": [1] * len(wp_ids),
                    "token_type_ids": [0] * len(wp_ids),
                }
                yield d

        features = datasets.Features(
            {
                **split.features,
                token_column: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "input_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_spans": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "attention_mask": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_type_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
            }
        )
        return datasets.Dataset.from_generator(inner, features=features, streaming=True)

    def run(
        self,
        dataset: DatasetDict,
        tokenizer: Lazy[Tokenizer],
        max_length: Optional[int] = None,
        token_column: str = "tokens",
    ) -> DatasetDict:
        tokenizer = tokenizer.construct()
        return DatasetDict({k: self._process_split(v, tokenizer, max_length, token_column) for k, v in dataset.items()})


@Step.register("loreiba.data.tokenize::train_tokenizer")
class TrainTokenizer(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, dataset: DatasetDict, model_path: str, vocab_size: Optional[int] = None) -> None:
        sentences = dataset["train"]["tokens"]
        if os.path.exists(model_path):
            self.logger.info(f"Already found model at {model_path}. Removing...")
            shutil.rmtree(model_path)
        train_tokenizer([" ".join(s) for s in sentences], model_path, vocab_size=vocab_size)
        # simple_train_tokenizer(sentences, model_path)
        self.logger.info(f"Wrote tokenizer to {model_path}")
