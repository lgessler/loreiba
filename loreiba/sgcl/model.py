import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from _socket import gethostname
from tango.common.exceptions import ConfigurationError
from tango.integrations.torch import Model, TrainCallback
from tango.integrations.transformers import Tokenizer
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, BertConfig, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from loreiba.sgcl.phrases.common import PhraseSgclConfig
from loreiba.sgcl.phrases.loss import phrase_guided_loss
from loreiba.sgcl.trees.common import TreeSgclConfig
from loreiba.sgcl.trees.loss import syntax_tree_guided_loss

logger = logging.getLogger(__name__)


@Model.register("loreiba.sgcl.model::sgcl_model")
class SGCLModel(Model):
    """
    Re-implementation of Syntax-Guided Contrastive Loss for Pre-trained Language Model
    (https://aclanthology.org/2022.findings-acl.191.pdf).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        tree_sgcl_config: Optional[TreeSgclConfig] = None,
        phrase_sgcl_config: Optional[PhraseSgclConfig] = None,
        pretrained_model_name_or_path: Optional[str] = None,
        bert_config: Dict[str, Any] = None,
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

        if pretrained_model_name_or_path is None and bert_config is None:
            raise ConfigurationError(f"Must provide either a pretrained model name or a BERT config.")

        config = BertConfig(
            **bert_config, vocab_size=len(tokenizer.get_vocab()), position_embedding_type="relative_key_query"
        )
        if pretrained_model_name_or_path is not None:
            logger.info(f"Initializing transformer stack from a pretrained model {pretrained_model_name_or_path}")
            self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)
        else:
            logger.info(f"Initializing a new BERT model with config {config}")
            self.encoder = BertModel(config=config, add_pooling_layer=False)
        self.lm_head = BertLMPredictionHead(config=config)
        self.pad_id = tokenizer.pad_token_id
        self.tree_sgcl_config = tree_sgcl_config
        self.phrase_sgcl_config = phrase_sgcl_config

    def _mlm_loss(self, preds, labels):
        if not (labels != -100).any():
            return 0.0
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        masked_lm_loss = loss_fct(preds.view(-1, self.encoder.config.vocab_size), labels.view(-1))
        return masked_lm_loss

    def forward(self, input_ids, token_type_ids, attention_mask, token_spans, head, deprel, labels=None, tree_sets=None, phrase_sets=None):
        encoder_outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
        hidden_states = encoder_outputs.hidden_states[1:]
        attentions = encoder_outputs.attentions
        x = encoder_outputs.last_hidden_state
        mlm_preds = self.lm_head(x)

        def log_metric(fname, value):
            if gethostname() == 'avi':
                with open(f"/tmp/{fname}", 'a') as f:
                    f.write(f"{value},")

        if labels is not None:
            outputs = {}
            mlm_loss = self._mlm_loss(mlm_preds, labels)
            loss = mlm_loss
            outputs["mlm_loss"] = mlm_loss
            if self.tree_sgcl_config is not None:
                tree_loss = syntax_tree_guided_loss(self.tree_sgcl_config, hidden_states, token_spans, tree_sets)
                loss += tree_loss
                print(f" tree_loss: {tree_loss:0.4f}", end="")
                log_metric("tree", tree_loss)
                outputs["tree_loss"] = tree_loss
            if self.phrase_sgcl_config is not None:
                phrase_loss = phrase_guided_loss(self.phrase_sgcl_config, attentions, attention_mask, phrase_sets)
                loss += phrase_loss
                print(f" phrase_loss: {phrase_loss:0.4f}", end="")
                log_metric("phrase", phrase_loss)
                outputs["phrase_loss"] = phrase_loss

            perplexity = mlm_loss.exp()
            print(f" perplexity: {perplexity:0.4f}", end="")
            log_metric("perplexity", perplexity)
            outputs["perplexity"] = perplexity

            print(end="\r")
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
