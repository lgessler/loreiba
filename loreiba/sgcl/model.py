import logging
import os
from typing import Any, Dict, Optional

import psutil
import torch
import torch.nn as nn
from _socket import gethostname
from tango.common import Registrable
from tango.common.exceptions import ConfigurationError
from tango.integrations.torch import Model, TrainCallback
from tango.integrations.transformers import Tokenizer
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, BertConfig, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from loreiba.sgcl.phrases.common import PhraseSgclConfig
from loreiba.sgcl.phrases.loss import phrase_guided_loss
from loreiba.sgcl.trees.common import TreeSgclConfig
from loreiba.sgcl.trees.loss import syntax_tree_guided_loss

logger = logging.getLogger(__name__)


class SgclEncoder(torch.nn.Module, Registrable):
    pass


@SgclEncoder.register("bert")
class BertEncoder(SgclEncoder):
    def __init__(self, tokenizer: Tokenizer, bert_config: Dict[str, Any]):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id
        config = BertConfig(
            **bert_config, vocab_size=len(tokenizer.get_vocab()), position_embedding_type="relative_key_query"
        )
        logger.info(f"Initializing a new BERT model with config {config}")
        self.config = config
        self.encoder = BertModel(config=config, add_pooling_layer=False)
        self.tokenizer = tokenizer

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def compute_loss(self, preds, labels):
        if not (labels != -100).any():
            return 0.0
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        masked_lm_loss = loss_fct(preds.view(-1, self.config.vocab_size), labels.view(-1))
        return masked_lm_loss

    @classmethod
    def construct_head(cls, encoder):
        return RobertaLMHead(config=encoder.config)


@Model.register("loreiba.sgcl.model::sgcl_model")
class SGCLModel(Model):
    """
    Re-implementation of Syntax-Guided Contrastive Loss for Pre-trained Language Model
    (https://aclanthology.org/2022.findings-acl.191.pdf).
    """

    def __init__(
        self,
        encoder: SgclEncoder,
        tree_sgcl_config: Optional[TreeSgclConfig] = None,
        phrase_sgcl_config: Optional[PhraseSgclConfig] = None,
        *args,
        **kwargs,
    ):
        """
        Provide `pretrained_model_name_or_path` if you want to use a pretrained model.
        Keep `bert_config` regardless as we need it for the LM head.

        Args:
            pretrained_model_name_or_path:
            bert_config:
            *args:
            **kwargs:
        """
        super().__init__()

        # a BERT-style Transformer encoder stack
        self.encoder = encoder
        self.head = encoder.__class__.construct_head(encoder)

        self.tree_sgcl_config = tree_sgcl_config
        self.phrase_sgcl_config = phrase_sgcl_config

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        token_spans,
        head,
        deprel,
        labels=None,
        tree_sets=None,
        phrase_sets=None,
    ):
        encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
        )
        hidden_states = encoder_outputs.hidden_states[1:]
        attentions = encoder_outputs.attentions
        last_encoder_state = encoder_outputs.last_hidden_state

        def log_metric(fname, value):
            if gethostname() == "avi":
                with open(f"/tmp/{fname}", "a") as f:
                    f.write(f"{value},")

        if labels is not None:
            outputs = {}
            preds = self.head(last_encoder_state)
            head_loss = self.encoder.compute_loss(preds, labels)
            loss = head_loss
            outputs["head_loss"] = head_loss
            outputs["progress_items"] = {
                "max_cuda_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "resident_memory_mb": psutil.Process().memory_info().rss / 1024**2,
                "head_loss": head_loss.item(),
            }
            if isinstance(self.encoder, BertEncoder):
                perplexity = head_loss.exp()
                outputs["progress_items"]["perplexity"] = perplexity.item()
            if self.training and self.tree_sgcl_config is not None:
                tree_loss = syntax_tree_guided_loss(self.tree_sgcl_config, hidden_states, token_spans, tree_sets)
                loss += tree_loss
                outputs["progress_items"]["tree_loss"] = tree_loss.item()
            if self.training and self.phrase_sgcl_config is not None:
                phrase_loss = phrase_guided_loss(self.phrase_sgcl_config, attentions, attention_mask, phrase_sets)
                loss += phrase_loss
                outputs["progress_items"]["phrase_loss"] = phrase_loss.item()

            outputs["loss"] = loss
            return outputs
        else:
            return {}


@TrainCallback.register("loreiba.model::write_model")
class WriteModelCallback(TrainCallback):
    def __init__(self, path: str, model_attr: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.model_attr = model_attr

    def post_train_loop(self, step: int, epoch: int) -> None:
        model = self.model
        if self.model_attr:
            model = getattr(model, self.model_attr)
        model.save_pretrained(self.path)
        self.logger.info(f"Wrote model to {self.path}")
