from typing import Any, Dict, List, Optional

import torch
from tango.common import Lazy
from tango.integrations.torch import DataCollator
from tango.integrations.transformers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling

from loreiba.sgcl.phrases.common import PhraseSgclConfig
from loreiba.sgcl.phrases.generation import generate_phrase_sets
from loreiba.sgcl.trees.common import TreeSgclConfig
from loreiba.sgcl.trees.generation import generate_subtrees


@DataCollator.register("loreiba.sgcl.collator::collator")
class SgclDataCollator(DataCollator):
    def __init__(
        self,
        tokenizer: Lazy[Tokenizer],
        static_masking: bool = True,
        text_field: str = "input_ids",
        span_field: str = "token_spans",
        tree_config: Optional[TreeSgclConfig] = None,
        phrase_config: Optional[PhraseSgclConfig] = None,
    ):
        tokenizer = tokenizer.construct()
        self.tokenizer = tokenizer
        self.text_pad_id = tokenizer.pad_token_id
        self.static_masking = static_masking
        self.text_field = text_field
        self.span_field = span_field
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        self.keys = None
        self.tree_config = tree_config
        self.phrase_config = phrase_config

    def __call__(self, batch) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        if self.keys is None:
            self.keys = list(batch[0].keys())

        output = {}
        for k in self.keys:
            output[k] = pad_sequence(
                (item[k] for item in batch),
                batch_first=True,
                padding_value=(0 if k != self.text_field else self.text_pad_id),
            )
            if k == "token_spans":
                output[k] = output[k].view(output[k].shape[0], -1, 2)

            if not self.static_masking and k == self.text_field:
                _, labels = self.mlm_collator.torch_mask_tokens(output[k])
                while (labels == -100).all():
                    _, labels = self.mlm_collator.torch_mask_tokens(output[k])
                output["labels"] = labels
        if self.tree_config is not None:
            output["tree_sets"] = generate_subtrees(self.tree_config, output["head"])
        if self.phrase_config is not None:
            output["phrase_sets"] = generate_phrase_sets(self.phrase_config, output["head"], output["token_spans"])

        return output
