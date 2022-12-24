from typing import Any, Dict, List, Optional

import torch
from tango.common import Lazy
from tango.integrations.torch import DataCollator
from tango.integrations.transformers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling

import loreiba.sgcl.trees as lst


@DataCollator.register("loreiba.sgcl.collator::collator")
class SgclDataCollator(DataCollator):
    def __init__(
        self,
        tokenizer: Lazy[Tokenizer],
        static_masking: bool = True,
        text_field: str = "input_ids",
        span_field: str = "token_spans",
    ):
        tokenizer = tokenizer.construct()
        self.tokenizer = tokenizer
        self.text_pad_id = tokenizer.pad_token_id
        self.static_masking = static_masking
        self.text_field = text_field
        self.span_field = span_field
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        self.keys = None

    def __call__(self, batch) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        if self.keys is None:
            self.keys = list(batch[0].keys())

        output = {}
        for k in self.keys:
            if k == self.span_field:
                max_len = max(item[k].shape[0] for item in batch)
                base = torch.full((len(batch), max_len, 2), -1, dtype=torch.long, device=batch[0][k].device)
                for i, item in enumerate(batch):
                    base[i, : item[k].shape[0]] = item[k]
                output[k] = base
            else:
                output[k] = pad_sequence(
                    (item[k] for item in batch),
                    batch_first=True,
                    padding_value=(0 if k != self.text_field else self.text_pad_id),
                )

            if not self.static_masking and k == self.text_field:
                input_ids, labels = self.mlm_collator.torch_mask_tokens(output[k])
                output["labels"] = labels

        return output
