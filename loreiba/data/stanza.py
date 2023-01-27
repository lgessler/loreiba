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

    output["dependency_token_spans"] = new_token_spans


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
            "dependency_token_spans": Sequence(feature=Value(dtype="int32", id=None), length=-1, id=None),
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

    def process_split(self, language_code, split, data, pipeline, batch_size, parsed_data_cache_dir, add_subword_edges):
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
            if add_subword_edges:
                extend_tree_with_subword_edges(output)

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
        add_subword_edges: bool = True,
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
                language_code, split, data, pipeline, batch_size, parsed_data_cache_dir, add_subword_edges
            )

        for split, dataset in dataset_dict.items():
            self.logger.info(f"Random {split} sentence: {random.choice(dataset_dict[split])}")

        return datasets.DatasetDict(dataset_dict)
