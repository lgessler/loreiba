import os
from pathlib import Path
from typing import Literal, Optional

import conllu
import datasets
import requests
from datasets import ClassLabel, DatasetDict, Sequence, Value
from github import Github
from tango import Step
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.transformers import Tokenizer


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
        shortcut: Optional[str] = None,
        conllu_path_train: Optional[str] = None,
        conllu_path_dev: Optional[str] = None,
    ) -> DatasetDict:
        def read_conllu(path):
            with open(path, "r") as f:
                return conllu.parse(f.read())

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
            [{"tokens": [t["form"] for t in s]} for d in train_docs for s in d],
            features=datasets.Features({"tokens": datasets.Sequence(datasets.Value(dtype="string"))}),
        )
        dev_dataset = datasets.Dataset.from_list(
            [{"tokens": [t["form"] for t in s]} for d in dev_docs for s in d],
            features=datasets.Features({"tokens": datasets.Sequence(datasets.Value(dtype="string"))}),
        )

        self.logger.info(f"First train sentence: {train_dataset[0]}")
        self.logger.info(f"First dev sentence: {dev_dataset[0]}")

        return DatasetDict({"train": train_dataset, "dev": dev_dataset})


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
                "upos": Sequence(
                    feature=ClassLabel(
                        names=[
                            "NOUN",
                            "PUNCT",
                            "ADP",
                            "NUM",
                            "SYM",
                            "SCONJ",
                            "ADJ",
                            "PART",
                            "DET",
                            "CCONJ",
                            "PROPN",
                            "PRON",
                            "X",
                            "_",
                            "ADV",
                            "INTJ",
                            "VERB",
                            "AUX",
                        ],
                        id=None,
                    ),
                    length=-1,
                    id=None,
                ),
                "xpos": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            }
        )

        train_dataset = datasets.Dataset.from_list([tokenlist_to_record(tl) for tl in train_conllu], features=features)
        dev_dataset = datasets.Dataset.from_list([tokenlist_to_record(tl) for tl in dev_conllu], features=features)
        test_dataset = datasets.Dataset.from_list([tokenlist_to_record(tl) for tl in test_conllu], features=features)
        self.logger.info(f"First train sentence: {train_dataset[0]}")
        self.logger.info(f"First dev sentence: {dev_dataset[0]}")

        return DatasetDict({"train": train_dataset, "dev": dev_dataset, "test": test_dataset})
