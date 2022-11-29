import time
from functools import partial
from torchmetrics import Accuracy

import datasets
import conllu
from typing import Dict, List, Any

import torch
from conllu import TokenList
from tango import Step, JsonFormat
from tango.common import Tqdm
import pandas as pd

from tango import Step
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.torch import EvalCallback, Model
from tango.integrations.transformers.tokenizer import Tokenizer
from tango.common.dataset_dict import DatasetDict, DatasetDictBase
from tango.integrations.transformers import DataCollator, Config

from transformers import AutoModelForSequenceClassification
from transformers.models.auto.auto_factory import _get_model_class
from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification



def read_conllu(path):
    with open(path, "r") as f:
        return conllu.parse(f.read())


def sentence_to_stype_instance(sentence: TokenList) -> Dict[str, Any]:
    stype = sentence.metadata["s_type"]
    text = sentence.metadata["text"]
    return {"label": stype, "text": text}


def conllu_to_stype_dataframe(conllu_path):
    records = [sentence_to_stype_instance(s) for s in read_conllu(conllu_path)]
    return pd.DataFrame.from_records(records)


@Model.register("demo_auto_model_wrapper::from_config", constructor="from_config")
class AutoModelForSequenceClassificationWrapper(AutoModelForSequenceClassification):
    @classmethod
    def from_config(cls, config: Config, **kwargs) -> Model:
        model_class = _get_model_class(config, cls._model_mapping)
        model = model_class._from_config(config, **kwargs)

        def forward(self, *args, **kwargs):
            output = model_class.forward(self, *args, **kwargs)
            labels = kwargs.pop("labels")
            acc = Accuracy().to(model.device)
            if labels is not None:
                preds = output.logits.max(1).indices.to(model.device)
                labels = labels.to(model.device)
                output = dict(output)
                output['accuracy'] = acc(preds, labels)
            return output

        model.forward = forward.__get__(model)
        return model



@Step.register("construct_stype_instances")
class ConstructStypeInstances(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(
        self,
        train_conllu: str,
        dev_conllu: str,
        test_conllu: str,
        tokenizer: Tokenizer,
        field_to_tokenize: str = "text",
        num_workers: int = 1,
    ) -> DatasetDict:
        label_mapping = {}

        def xform_fn(example: Dict[str, Any]) -> Dict[str, Any]:
            tokenizer_output = tokenizer(example[field_to_tokenize])
            output = {"label": label_mapping[example['label']], **tokenizer_output}
            return output

        raw_dataset = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_pandas(conllu_to_stype_dataframe(train_conllu)),
                "dev": datasets.Dataset.from_pandas(conllu_to_stype_dataframe(dev_conllu)),
                "test": datasets.Dataset.from_pandas(conllu_to_stype_dataframe(test_conllu)),
            }
        )

        labels = set(i['label'] for i in raw_dataset['dev'])
        for i, v in enumerate(sorted(list(labels))):
            label_mapping[v] = i
        print(label_mapping)

        processed_dataset = raw_dataset.map(xform_fn, batched=False, num_proc=num_workers, with_indices=False, remove_columns=["text"])
        self.logger.info(processed_dataset["train"][0])
        return processed_dataset
