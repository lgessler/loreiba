from typing import Any, Dict

from tango.integrations.torch import DataCollator
from tango.integrations.transformers import Tokenizer


@DataCollator.register("loreiba.sgcl.collator::collator")
class SgclDataCollator(DataCollator):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items) -> Dict[str, Any]:
        padded = {}
        for key in items[0].keys():
            max_length = max(len(item[key]) for item in items)
            padded[key] = [(item[key] + ([0] * (max_length - len(item[key])))) for item in items]

        return padded
