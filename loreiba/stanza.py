import datasets
import more_itertools
import stanza
from datasets import ClassLabel, Dataset, DatasetDict, Sequence, Value
from tango import Step
from tango.common import Tqdm
from tango.integrations.datasets import DatasetsFormat


@Step.register("loreiba.stanza::stanza_parse_dataset")
class StanzaParseDataset(Step):
    DETERMINISTIC = False  # assume that parsers are non-deterministic :(
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(
        self,
        dataset: DatasetDict,
        language_code: str,
        batch_size: int = 32,
        allow_retokenization: bool = True,
    ) -> DatasetDict:
        config = {
            "processors": "tokenize,mwt,pos,lemma,depparse",
            "lang": language_code,
            "use_gpu": True,
            "logging_level": "INFO",
            "tokenize_pretokenized": not allow_retokenization,
        }
        pipeline = stanza.Pipeline(**config)

        dataset_dict = {}

        def sentence_to_record(s):
            # filter out supertokens
            s = [t for t in s if isinstance(t["id"], int)]
            return {
                "tokens": [t["text"] for t in s],
                "lemmas": [t["lemma"] for t in s],
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
                "feats": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "head": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "deprel": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            }
        )

        for split, data in dataset.items():
            space_separated = [" ".join(ts) for ts in data["tokens"]]
            chunks = list(more_itertools.chunked(space_separated, batch_size))

            outputs = []
            for chunk in Tqdm.tqdm(chunks, desc=f"Parsing split {split}..."):
                output = pipeline("\n\n".join(chunk))
                for sentence in output.sentences:
                    s = sentence.to_dict()
                    outputs.append(sentence_to_record(s))
            dataset_dict[split] = datasets.Dataset.from_list(outputs, features=features)
            self.logger.info(f"Finished processing {split}")

        for split, dataset in dataset_dict.items():
            self.logger.info(f"First {split} sentence: {dataset_dict[split][0]}")

        return datasets.DatasetDict(dataset_dict)
