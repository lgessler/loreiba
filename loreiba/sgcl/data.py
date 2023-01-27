from itertools import chain, repeat

import datasets
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Sequence, Value
from tango import Step
from tango.common import Lazy, Tqdm
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.transformers import Tokenizer
from transformers import DataCollatorForLanguageModeling


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return chain.from_iterable(repeat(tuple(iterable), n))


@Step.register("loreiba.sgcl.data::finalize")
class Finalize(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    # Note: we are expecting "full conllu" in the dataset argument here, as would be produced by
    # loreiba.data.stanza::stanza_parse_dataset
    def run(
        self,
        dataset: DatasetDict,
        tokenizer: Lazy[Tokenizer],
        static_masking: bool = True,
    ) -> DatasetDict:
        tokenizer = tokenizer.construct()
        dataset = dataset.remove_columns(["tokens", "lemmas", "upos"])
        # It's OK to peek at test since UD deprels are a fixed set--this is just for convenience, not cheating
        deprels = sorted(
            list(
                set(d for s in dataset["train"]["deprel"] for d in s)
                | set(d for s in dataset["dev"]["deprel"] for d in s)
            )
        )
        self.logger.info(f"Using deprel set: {deprels}")
        xpos = sorted(
            list(
                set(x for s in dataset["train"]["xpos"] for x in s) | set(x for s in dataset["dev"]["xpos"] for x in s)
            )
        )
        self.logger.info(f"Using xpos set: {xpos}")

        features = datasets.Features(
            {
                "input_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "attention_mask": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_type_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "token_spans": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
                "head": Sequence(feature=Value(dtype="int16", id=None), length=-1, id=None),
                "deprel": Sequence(feature=ClassLabel(names=deprels, id=None), length=-1, id=None),
                "xpos": Sequence(feature=ClassLabel(names=xpos, id=None), length=-1, id=None),
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
                        "token_type_ids": v["token_type_ids"],
                        "attention_mask": v["attention_mask"],
                        "token_spans": v["token_spans"],
                        "head": [int(i) for i in v["head"]],
                        "deprel": v["deprel"],
                        "xpos": v["xpos"],
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
