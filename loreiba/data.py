import dataclasses
import os
import random
import shutil
import sys
from itertools import chain, repeat
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

import conllu
import datasets
import more_itertools as mit
import requests
import stanza
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Sequence, Value
from github import Github
from tango import Step
from tango.common import Lazy, Tqdm
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.transformers import Tokenizer
from transformers import DataCollatorForLanguageModeling

from loreiba.tokenizers import simple_train_tokenizer


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return chain.from_iterable(repeat(tuple(iterable), n))


@Step.register("loreiba.data::read_text_only_conllu")
class ReadTextOnlyConllu(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    PATH_MAP = {
        "coptic": "data/coptic/converted",
        "greek": "data/greek/converted",
        "indonesian": "data/indonesian/converted_punct",
        "maltese": "data/maltese/converted_punct",
        "tamil": "data/tamil/converted_punct",
        "uyghur": "data/uyghur/converted_punct",
        "wolof": "data/wolof/converted_punct",
    }

    def run(
        self,
        stanza_retokenize: bool = False,
        stanza_language_code: Optional[str] = None,
        shortcut: Optional[str] = None,
        conllu_path_train: Optional[str] = None,
        conllu_path_dev: Optional[str] = None,
        stanza_use_mwt: bool = True,
    ) -> DatasetDict:
        if stanza_retokenize:
            config = {
                "processors": "tokenize,mwt" if stanza_use_mwt else "tokenize",
                "lang": stanza_language_code,
                "use_gpu": True,
                "logging_level": "INFO",
                "tokenize_pretokenized": False,
                "tokenize_no_ssplit": True,
            }
            pipeline = stanza.Pipeline(**config)

        def retokenize(sentences, path):
            batch_size = 256

            space_separated = [" ".join(ts) for ts in sentences]
            chunks = list(mit.chunked(space_separated, batch_size))

            outputs = []
            for chunk in Tqdm.tqdm(chunks, desc=f"Retokenizing {path} with Stanza..."):
                output = pipeline("\n\n".join(chunk))
                for sentence in output.sentences:
                    s = sentence.to_dict()
                    retokenized = [t["text"] for t in s]
                    outputs.append(retokenized)
            for old, new in zip(sentences, outputs):
                if len(old) != len(new):
                    self.logger.debug(f"Retokenized sentence from {len(old)} to {len(new)}:\n\t{old}\n\t{new}\n")
            return outputs

        def read_conllu(path):
            with open(path, "r") as f:
                sentences = [[t["form"] for t in s] for s in conllu.parse(f.read())]
                if stanza_retokenize:
                    sentences = retokenize(sentences, path)
                return sentences

        if shortcut is not None:
            if shortcut not in ReadTextOnlyConllu.PATH_MAP:
                raise ValueError(f"Unrecognized shortcut: {shortcut}")
            self.logger.info(f"Recognized shortcut {shortcut}")
            conllu_path_train = ReadTextOnlyConllu.PATH_MAP[shortcut] + os.sep + "train"
            conllu_path_dev = ReadTextOnlyConllu.PATH_MAP[shortcut] + os.sep + "dev"
            self.logger.info(f"Train path set to {conllu_path_train}")
            self.logger.info(f"Dev path set to {conllu_path_dev}")

        train_docs = (
            [read_conllu(conllu_path_train)]
            if conllu_path_train.endswith(".conllu")
            else [read_conllu(f) for f in Path(conllu_path_train).glob("**/*.conllu") if f.is_file()]
        )
        self.logger.info(
            f"Loaded {len(train_docs)} training docs from {conllu_path_train} "
            f"containing {len([t for d in train_docs for s in d for t in s])} tokens."
        )
        dev_docs = (
            [read_conllu(conllu_path_dev)]
            if conllu_path_dev.endswith(".conllu")
            else [read_conllu(f) for f in Path(conllu_path_dev).glob("**/*.conllu") if f.is_file()]
        )
        self.logger.info(
            f"Loaded {len(dev_docs)} dev docs from {conllu_path_dev} "
            f"containing {len([t for d in dev_docs for s in d for t in s])} tokens."
        )

        train_dataset = datasets.Dataset.from_list(
            [{"tokens": s} for d in train_docs for s in d],
            features=datasets.Features({"tokens": datasets.Sequence(datasets.Value(dtype="string"))}),
        )
        dev_dataset = datasets.Dataset.from_list(
            [{"tokens": s} for d in dev_docs for s in d],
            features=datasets.Features({"tokens": datasets.Sequence(datasets.Value(dtype="string"))}),
        )

        self.logger.info(f"First train sentence: {train_dataset[0]}")
        self.logger.info(f"First dev sentence: {dev_dataset[0]}")

        return DatasetDict({"train": train_dataset, "dev": dev_dataset})


@Step.register("loreiba.data::tokenize_plus")
class TokenizePlus(Step):
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
                offsets.append((len(tokens), len(tokens) + len(wp_ids) - 1))
            else:
                tokens.append(tokenizer.unk_token_id)
                offsets.append((len(tokens), len(tokens)))
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
        sentences = split[token_column]
        output = []
        for sentence in sentences:
            wp_ids, token_spans = self.intra_word_tokenize(sentence, tokenizer, max_length)
            flattened = []
            for pair in token_spans:
                flattened.extend(pair)
            d = {"input_ids": wp_ids, "token_spans": flattened, token_column: sentence[: len(token_spans) - 2]}
            output.append(d)

        features = datasets.Features(
            {
                token_column: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "input_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_spans": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
            }
        )
        return datasets.Dataset.from_list(output, features=features)

    def run(
        self,
        dataset: DatasetDict,
        tokenizer: Lazy[Tokenizer],
        max_length: Optional[int] = None,
        token_column: str = "tokens",
    ) -> DatasetDict:
        tokenizer = tokenizer.construct()
        return DatasetDict({k: self._process_split(v, tokenizer, max_length, token_column) for k, v in dataset.items()})


@Step.register("loreiba.data::train_tokenizer")
class TrainTokenizer(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, dataset: DatasetDict, model_path: str) -> None:
        sentences = dataset["train"]["tokens"]
        if os.path.exists(model_path):
            self.logger.info(f"Already found model at {model_path}. Removing...")
            shutil.rmtree(model_path)
        simple_train_tokenizer(sentences, model_path)
        self.logger.info(f"Wrote tokenizer to {model_path}")


@Step.register("loreiba.data::read_ud_treebank")
class ReadUDTreebank(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    REPO_MAP = {
        "coptic": "UD_Coptic-Scriptorium",
        "greek": "UD_Ancient_Greek-PROIEL",
        "indonesian": "UD_Indonesian-GSD",
        "maltese": "UD_Maltese-MUDT",
        "tamil": "UD_Tamil-TTB",
        "uyghur": "UD_Uyghur-UDT",
        "wolof": "UD_Wolof-WTB",
    }

    def run(
        self,
        shortcut: Optional[str] = None,
        repo: Optional[str] = None,
        tag: str = "r2.11",
    ) -> DatasetDict:
        api = Github()
        if shortcut is not None:
            if shortcut not in ReadUDTreebank.REPO_MAP:
                raise ValueError(f"Unrecognized shortcut: {shortcut}")
            repo = ReadUDTreebank.REPO_MAP[shortcut]
        r = api.get_repo(f"UniversalDependencies/{repo}")
        all_tags = r.get_tags()
        filtered_tags = [t for t in all_tags if t.name == tag]
        if len(filtered_tags) == 0:
            raise ValueError(f"Requested tag {tag} was not found. Available tags: {all_tags}")

        files = [
            (f.path, f.download_url)
            for f in r.get_contents("/", ref=filtered_tags[0].commit.sha)
            if f.path.endswith(".conllu")
        ]
        if len(files) != 3:
            raise ValueError("Repositories without a train, dev, and test split are not supported.")
        train_url = [url for name, url in files if "train.conllu" in name][0]
        dev_url = [url for name, url in files if "dev.conllu" in name][0]
        test_url = [url for name, url in files if "test.conllu" in name][0]

        train_conllu = conllu.parse(requests.get(train_url).text)
        dev_conllu = conllu.parse(requests.get(dev_url).text)
        test_conllu = conllu.parse(requests.get(test_url).text)

        def tokenlist_to_record(tl: conllu.TokenList):
            return {
                "idx": tl.metadata["sent_id"],
                "text": tl.metadata["text"],
                "tokens": [t["form"] for t in tl],
                "lemmas": [t["lemma"] for t in tl],
                "upos": [t["upos"] for t in tl],
                "xpos": [t["xpos"] for t in tl],
                "feats": [t["feats"] for t in tl],
                "head": [t["head"] for t in tl],
                "deprel": [t["deprel"] for t in tl],
                "deps": [t["deps"] for t in tl],
                "misc": [t["misc"] for t in tl],
            }

        features = datasets.Features(
            {
                "deprel": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "deps": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "feats": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "head": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "idx": Value(dtype="string", id=None),
                "lemmas": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "misc": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "text": Value(dtype="string", id=None),
                "tokens": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "upos": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "xpos": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            }
        )

        train_dataset = datasets.Dataset.from_list([tokenlist_to_record(tl) for tl in train_conllu], features=features)
        dev_dataset = datasets.Dataset.from_list([tokenlist_to_record(tl) for tl in dev_conllu], features=features)
        test_dataset = datasets.Dataset.from_list([tokenlist_to_record(tl) for tl in test_conllu], features=features)
        self.logger.info(f"First train sentence: {train_dataset[0]}")
        self.logger.info(f"First dev sentence: {dev_dataset[0]}")

        return DatasetDict({"train": train_dataset, "dev": dev_dataset, "test": test_dataset})


def extend_tree_with_subword_edges(output):
    token_spans = output["token_spans"]
    head = output["head"]
    deprel = output["deprel"]
    orig_head = head.copy()
    orig_deprel = deprel.copy()

    # Map from old token IDs to new token IDs after subword expansions
    id_map = {}
    running_diff = 0
    assert len(token_spans) // 2 == len(head) + 2
    for token_id in range(0, len(token_spans) // 2):
        id_map[token_id] = token_id + running_diff
        b, e = token_spans[token_id * 2 : (token_id + 1) * 2]
        running_diff += e - b

    # Note how many subwords have been added so far
    new_token_spans = []
    for token_id in range(0, len(token_spans) // 2):
        # Inclusive indices of the subwords that the original token corresponds to
        b, e = token_spans[token_id * 2 : (token_id + 1) * 2]

        # If not a special token (we're assuming there are 2 on either end of the sequence),
        # replace the head value of the current token with the mapped value
        if token_id != 0 and token_id != ((len(token_spans) // 2) - 1):
            head[id_map[token_id] - 1] = id_map[orig_head[token_id - 1]]
        if e == b:
            # If we have a token that corresponds to a single subword, just append the same token_spans values
            new_token_spans.append(b)
            new_token_spans.append(e)
        else:
            # Note how many expansion subwords we'll add
            diff = e - b
            # This is the first subword in the token's index into head and deprel. Remember token_id is 1-indexed
            first_subword_index = id_map[token_id] - 1
            new_token_spans.append(b)
            new_token_spans.append(b)
            # For each expansion subword, add a separate token_spans entry and expand head and deprel.
            # Head's value is the ID of the first subword in the token it belongs to
            for j in range(1, diff + 1):
                new_token_spans.append(b + j)
                new_token_spans.append(b + j)
                head.insert(first_subword_index + j, id_map[token_id])
                deprel.insert(first_subword_index + j, "subword")

    heads = [int(x) for x in head]
    for h in heads:
        current = h
        seen = set()
        while current != 0:
            if current in seen:
                raise Exception(f"Cycle detected!\n{orig_head}\n{orig_deprel}\n{token_spans}\n\n{head}\n{deprel}")
            seen.add(current)
            current = heads[current - 1]

    output["token_spans"] = new_token_spans


@Step.register("loreiba.data::stanza_parse_dataset")
class StanzaParseDataset(Step):
    DETERMINISTIC = True  # actually we should assume parsers are non-deterministic, but we don't care
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(
        self,
        dataset: DatasetDict,
        language_code: str,
        batch_size: int = 32,
        allow_retokenization: bool = True,
        stanza_use_mwt: bool = True,
        add_subword_edges: bool = True,
    ) -> DatasetDict:
        config = {
            "processors": "tokenize,mwt,pos,lemma,depparse" if stanza_use_mwt else "tokenize,pos,lemma,depparse",
            "lang": language_code,
            "use_gpu": True,
            "logging_level": "INFO",
            "tokenize_pretokenized": not allow_retokenization,
            "tokenize_no_ssplit": True,  # never allow sentence resegmentation
        }
        pipeline = stanza.Pipeline(**config)

        dataset_dict = {}

        def sentence_to_record(s, orig):
            # filter out supertokens
            s = [t for t in s if isinstance(t["id"], int)]
            if len(orig) != len([t["text"] for t in s]):
                print(orig, len(orig), file=sys.stderr)
                print([t["text"] for t in s], len([t["text"] for t in s]), file=sys.stderr)
                print([t["head"] for t in s], len([t["head"] for t in s]), file=sys.stderr)
                raise Exception("Stanza returned a different number of tokens than we started with!")
            return {
                "tokens": [t["text"] for t in s],
                "lemmas": [t.get("lemma", "") for t in s],
                "upos": [t["upos"] for t in s],
                "xpos": [t["xpos"] for t in s],
                "feats": [t.get("feats", "") for t in s],  # for some reason, feats may not appear
                "head": [t["head"] for t in s],
                "deprel": [t["deprel"] for t in s],
            }

        features = datasets.Features(
            {
                "lemmas": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "tokens": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "upos": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "xpos": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "feats": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "head": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "deprel": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "input_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_spans": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
            }
        )

        for split, data in dataset.items():
            chunks = list(mit.chunked(data["tokens"], batch_size))
            outputs = []
            for chunk in Tqdm.tqdm(chunks, desc=f"Parsing split {split}..."):
                inputs = [stanza.Document([], text=[sentence for sentence in chunk])]
                output = pipeline(inputs)[0].to_dict()
                for i, sentence in enumerate(output):
                    record = sentence_to_record(sentence, chunk[i])
                    outputs.append(record)

            for i, output in enumerate(outputs):
                output["input_ids"] = data[i]["input_ids"]
                output["token_spans"] = data[i]["token_spans"]
                if add_subword_edges:
                    extend_tree_with_subword_edges(output)

            dataset_dict[split] = datasets.Dataset.from_list(outputs, features=features)
            self.logger.info(f"Finished processing {split}")

        for split, dataset in dataset_dict.items():
            self.logger.info(f"Random {split} sentence: {random.choice(dataset_dict[split])}")

        return datasets.DatasetDict(dataset_dict)


@Step.register("loreiba.data::finalize")
class Finalize(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(
        self,
        dataset: DatasetDict,
        tokenizer: Lazy[Tokenizer],
        static_masking: bool = True,
    ) -> DatasetDict:
        tokenizer = tokenizer.construct()
        dataset = dataset.remove_columns(["tokens", "lemmas", "xpos", "upos"])
        # It's OK to peek at test since UD deprels are a fixed set--this is just for convenience, not cheating
        deprels = sorted(
            list(
                set(d for s in dataset["train"]["deprel"] for d in s)
                | set(d for s in dataset["dev"]["deprel"] for d in s)
            )
        )
        self.logger.info(f"Using deprel set: {deprels}")

        features = datasets.Features(
            {
                "input_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_spans": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "head": Sequence(feature=Value(dtype="int16", id=None), length=-1, id=None),
                "deprel": Sequence(feature=ClassLabel(names=deprels, id=None), length=-1, id=None),
            }
        )

        mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        if static_masking:
            features["labels"] = Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None)

        new_dataset = {}
        for split, rows in dataset.items():

            def get_rows():
                for v in Tqdm.tqdm(
                    ncycles(rows, 10) if static_masking else rows,
                    desc=f"Constructing {split}..",
                    total=len(rows) * (10 if static_masking else 1),
                ):
                    new_row = {
                        "input_ids": v["input_ids"],
                        "token_spans": v["token_spans"],
                        "head": [int(i) for i in v["head"]],
                        "deprel": v["deprel"],
                    }
                    if static_masking:
                        _, labels = mlm_collator.torch_mask_tokens(torch.tensor([new_row["input_ids"]]))
                        new_row["labels"] = labels[0].tolist()
                    yield new_row

            if static_masking:
                # needed to do this to avoid loading the 10x dataset into memory
                dummy = rows[0].copy()
                del dummy["feats"]
                if static_masking:
                    _, labels = mlm_collator.torch_mask_tokens(torch.tensor([dummy["input_ids"]]))
                    dummy["labels"] = labels[0].tolist()
                masked = Dataset.from_list([dummy], features=features)
                for item in get_rows():
                    masked = masked.add_item(item)
                new_dataset[split] = masked
            else:
                new_dataset[split] = Dataset.from_list(list(get_rows()), features=features)
            self.logger.info(f"Appended split with {len(new_dataset[split])} sequences")

        return datasets.DatasetDict(new_dataset).with_format("torch")


# feature=ClassLabel(
#     names=[
#         "NOUN",
#         "PUNCT",
#         "ADP",
#         "NUM",
#         "SYM",
#         "SCONJ",
#         "ADJ",
#         "PART",
#         "DET",
#         "CCONJ",
#         "PROPN",
#         "PRON",
#         "X",
#         "_",
#         "ADV",
#         "INTJ",
#         "VERB",
#         "AUX",
#     ],
#     id=None,
# ),
