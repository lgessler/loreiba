import os
import random
import sys

import datasets
import more_itertools as mit
import stanza
from datasets import DatasetDict, Sequence, Value
from tango import Step
from tango.common import Tqdm
from tango.integrations.datasets import DatasetsFormat

from loreiba.common import dill_dump, dill_load


@Step.register("loreiba.data.stanza::stanza_parse_dataset")
class StanzaParseDataset(Step):
    DETERMINISTIC = True  # actually we should assume parsers are non-deterministic, but we don't care
    CACHEABLE = True
    FORMAT = DatasetsFormat()

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
            "attention_mask": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
            "token_type_ids": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
            "token_spans": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
        }
    )

    @staticmethod
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

    def process_split(self, language_code, split, data, pipeline, batch_size, parsed_data_cache_dir):
        if parsed_data_cache_dir is not None:
            cache_key = os.path.join(parsed_data_cache_dir, f"{language_code}-{split}.dill")
            if os.path.exists(cache_key):
                self.logger.info(f"Found parsed outputs at {cache_key}")
                return dill_load(cache_key)

        chunks = list(mit.chunked(data["tokens"], batch_size))
        outputs = []
        for chunk in Tqdm.tqdm(chunks, desc=f"Parsing split {split}..."):
            inputs = [stanza.Document([], text=[sentence for sentence in chunk])]
            output = pipeline(inputs)[0].to_dict()
            for i, sentence in enumerate(output):
                record = StanzaParseDataset.sentence_to_record(sentence, chunk[i])
                outputs.append(record)

        for i, output in enumerate(outputs):
            output["input_ids"] = data[i]["input_ids"]
            output["token_type_ids"] = data[i]["token_type_ids"]
            output["attention_mask"] = data[i]["attention_mask"]
            output["token_spans"] = data[i]["token_spans"]

        self.logger.info(f"Finished processing {split}")
        instances = datasets.Dataset.from_list(outputs, features=StanzaParseDataset.features)
        if parsed_data_cache_dir is not None:
            cache_key = os.path.join(parsed_data_cache_dir, f"{language_code}-{split}.dill")
            self.logger.info(f"Writing parsed outputs to cache at {cache_key}")
            os.makedirs(parsed_data_cache_dir, exist_ok=True)
            dill_dump(instances, cache_key)

        return instances

    def run(
        self,
        dataset: DatasetDict,
        language_code: str,
        batch_size: int = 32,
        allow_retokenization: bool = True,
        stanza_use_mwt: bool = True,
        parsed_data_cache_dir: str = "./workspace/parsed",
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
        for split, data in dataset.items():
            dataset_dict[split] = self.process_split(
                language_code, split, data, pipeline, batch_size, parsed_data_cache_dir
            )

        for split, dataset in dataset_dict.items():
            self.logger.info(f"Random {split} sentence: {random.choice(dataset_dict[split])}")

        return datasets.DatasetDict(dataset_dict)
