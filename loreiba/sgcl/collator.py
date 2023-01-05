from typing import Any, Dict, List, Optional, Tuple

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


# copied from HF implementation, but does not do the 10% [UNK] and 10% random word replacement
def torch_mask_tokens(
    inputs: Any, tokenizer, special_tokens_mask: Optional[Any] = None, mlm_probability: float = 0.15
) -> Tuple[Any, Any]:
    import torch

    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # # 10% of the time, we replace masked input tokens with random word
    # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    # random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    # inputs[indices_random] = random_words[indices_random]

    # # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


@DataCollator.register("loreiba.sgcl.collator::collator")
class SgclDataCollator(DataCollator):
    def __init__(
        self,
        tokenizer: Lazy[Tokenizer],
        mask_only: bool = False,
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
        self.mask_only = mask_only
        if not mask_only:
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
                if not self.mask_only:
                    _, labels = self.mlm_collator.torch_mask_tokens(output[k])
                else:
                    _, labels = torch_mask_tokens(output[k], self.tokenizer)
                while (labels == -100).all():
                    if not self.mask_only:
                        _, labels = self.mlm_collator.torch_mask_tokens(output[k])
                    else:
                        _, labels = torch_mask_tokens(output[k], self.tokenizer)
                output["labels"] = labels

        if self.tree_config is not None:
            output["tree_sets"] = generate_subtrees(self.tree_config, output["head"])
        if self.phrase_config is not None:
            output["phrase_sets"] = generate_phrase_sets(self.phrase_config, output["head"], output["token_spans"])

        return output
