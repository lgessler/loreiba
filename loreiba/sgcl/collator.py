from typing import Any, Dict, List, Optional

import torch
from tango.common import Lazy
from tango.integrations.torch import DataCollator
from tango.integrations.transformers import Tokenizer
from transformers import DataCollatorForLanguageModeling


@DataCollator.register("loreiba.sgcl.collator::collator")
class SgclDataCollator(DataCollator):
    def __init__(self, tokenizer: Lazy[Tokenizer], text_fields: List[str] = ()):
        tokenizer = tokenizer.construct()
        self.tokenizer = tokenizer
        self.text_pad_id = tokenizer.pad_token_id
        self.text_fields = text_fields
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    def __call__(self, batch) -> Dict[str, Any]:
        padded = {}
        for key in batch[0].keys():
            max_length = max(len(item[key]) for item in batch)
            pad_id = 0 if key not in self.text_fields else self.text_pad_id
            value = torch.vstack(
                [
                    torch.hstack(
                        (
                            item[key],
                            torch.tensor([pad_id] * (max_length - len(item[key])), dtype=torch.long).to(
                                item[key].device
                            ),
                        )
                    )
                    for item in batch
                ]
            ).to(batch[0][key].device)
            if key == "input_ids":
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
