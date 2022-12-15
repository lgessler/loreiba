from typing import Any, Dict, List, Optional

import torch
from tango.common import Lazy
from tango.integrations.torch import DataCollator
from tango.integrations.transformers import Tokenizer
from transformers import DataCollatorForLanguageModeling

import loreiba.sgcl.trees as lst


@DataCollator.register("loreiba.sgcl.collator::collator")
class SgclDataCollator(DataCollator):
    def __init__(self, tokenizer: Lazy[Tokenizer], text_field: str = "input_ids", span_field: str = "token_spans"):
        tokenizer = tokenizer.construct()
        self.tokenizer = tokenizer
        self.text_pad_id = tokenizer.pad_token_id
        self.text_field = text_field
        self.span_field = span_field
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    def __call__(self, batch) -> Dict[str, Any]:
        padded = {}
        for key in batch[0].keys():
            max_length = max(len(item[key]) for item in batch)
            pad = [self.text_pad_id] if key == self.text_field else [[-1, -1]] if key == self.span_field else [0]
            if key == self.span_field:
                seq_values = [
                    torch.vstack(
                        (
                            item[key],
                            torch.tensor(pad * (max_length - len(item[key])), dtype=torch.long).to(item[key].device),
                        ),
                    )
                    if (max_length - len(item[key])) > 0
                    else item[key]
                    for item in batch
                ]
                value = torch.stack(seq_values, dim=0).to(batch[0][key].device)
            else:
                seq_values = [
                    torch.hstack(
                        (
                            item[key],
                            torch.tensor(pad * (max_length - len(item[key])), dtype=torch.long).to(item[key].device),
                        )
                    )
                    for item in batch
                ]
                value = torch.vstack(seq_values).to(batch[0][key].device)
            if key == self.text_field:
                input_ids, labels = self.mlm_collator.torch_mask_tokens(value)
                num_masked = (input_ids.view(-1) == self.tokenizer.mask_token_id).sum().item()
                while num_masked == 0:
                    input_ids, labels = self.mlm_collator.torch_mask_tokens(value)
                    num_masked = (input_ids.view(-1) == self.tokenizer.mask_token_id).sum().item()
                padded[key] = input_ids
                padded["labels"] = labels
            else:
                padded[key] = value

        return padded
