from typing import Any, Dict, List, Optional

import torch
from tango.common import Lazy
from tango.integrations.torch import DataCollator
from tango.integrations.transformers import Tokenizer


@DataCollator.register("loreiba.sgcl.collator::collator")
class SgclDataCollator(DataCollator):
    def __init__(self, tokenizer: Lazy[Tokenizer], text_fields: List[str] = ()):
        tokenizer = tokenizer.construct()
        self.tokenizer = tokenizer
        self.text_pad_id = tokenizer.pad_token_id
        self.text_fields = text_fields

    def __call__(self, items) -> Dict[str, Any]:
        padded = {}
        for key in items[0].keys():
            max_length = max(len(item[key]) for item in items)
            pad_id = 0 if key not in self.text_fields else self.text_pad_id
            padded[key] = [
                torch.hstack((item[key], torch.tensor([pad_id] * (max_length - len(item[key]))).to(item[key].device)))
                for item in items
            ]

        return padded
