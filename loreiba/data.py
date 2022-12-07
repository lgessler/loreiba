import os
import random
from pathlib import Path
from typing import Literal, Optional

import conllu
import datasets
import more_itertools as mit
import requests
import stanza
from datasets import ClassLabel, Dataset, DatasetDict, Sequence, Value
from github import Github
from tango import Step
from tango.common import Lazy, Tqdm
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.transformers import Tokenizer

from loreiba.tokenizers import simple_train_tokenizer


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

    def _process_split(
        self, split: Dataset, tokenizer: Tokenizer, max_length: Optional[int], token_column: str
    ) -> Dataset:
        sentences = split[token_column]
        output = []
        for sentence in sentences:
            input_str = " ".join(sentence)
            r = tokenizer.encode_plus(
                input_str,
                add_special_tokens=True,
                return_token_type_ids=True,
                return_length=True,
                return_attention_mask=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            # b, e are substring indices into the input_str showing how the wordpiece at index i
            # corresponds to a substring in the input. Special tokens are indexed at (0,0).
            # We can calculate which token index a wordpiece corresponds to by counting
            # the number of spaces to the left. Note that these are 0 indexes and special tokens
            # are indexed as corresponding to token index -1
            r["token_indexes"] = [input_str[:b].count(" ") if b != e else -1 for b, e in r["offset_mapping"]]
            del r["offset_mapping"]
            del r["length"]
            r[token_column] = sentence
            output.append(dict(r))

        features = datasets.Features(
            {
                token_column: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "input_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_type_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "attention_mask": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_indexes": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
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

        def sentence_to_record(s):
            # filter out supertokens
            s = [t for t in s if isinstance(t["id"], int)]
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
                "token_type_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "attention_mask": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_indexes": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
            }
        )

        for split, data in dataset.items():
            space_separated = [" ".join(ts) for ts in data["tokens"]]
            chunks = list(mit.chunked(space_separated, batch_size))

            outputs = []
            for chunk in Tqdm.tqdm(chunks, desc=f"Parsing split {split}..."):
                output = pipeline("\n\n".join(chunk))
                for sentence in output.sentences:
                    s = sentence.to_dict()
                    record = sentence_to_record(s)
                    outputs.append(record)

            for i, output in enumerate(outputs):
                output["input_ids"] = data[i]["input_ids"]
                output["token_type_ids"] = data[i]["token_type_ids"]
                output["attention_mask"] = data[i]["attention_mask"]
                output["token_indexes"] = data[i]["token_indexes"]

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
    ) -> DatasetDict:
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
                "token_type_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "attention_mask": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_indexes": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "head": Sequence(feature=Value(dtype="int16", id=None), length=-1, id=None),
                "deprel": Sequence(feature=ClassLabel(names=deprels, id=None), length=-1, id=None),
            }
        )

        new_dataset = {}
        for split, rows in dataset.items():
            new_dataset[split] = Dataset.from_list(
                [
                    {
                        "input_ids": v["input_ids"],
                        "token_type_ids": v["token_type_ids"],
                        "attention_mask": v["attention_mask"],
                        "token_indexes": v["token_indexes"],
                        "head": [int(i) for i in v["head"]],
                        "deprel": v["deprel"],
                    }
                    for v in rows
                ],
                features=features,
            )

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
