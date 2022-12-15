import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tango.common.exceptions import ConfigurationError
from tango.integrations.torch import Model, TrainCallback
from tango.integrations.transformers import Config, Tokenizer
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, BertConfig, RobertaConfig, RobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from loreiba.sgcl.trees import TreeSgclConfig, syntax_tree_guided_loss

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
        sgcl_config: TreeSgclConfig,
        pretrained_model_name_or_path: Optional[str] = None,
        roberta_config: Dict[str, Any] = None,
        *args,
        **kwargs,
    ):
        """
        Provide `pretrained_model_name_or_path` if you want to use a pretrained model.
        Keep `roberta_config` regardless as we need it for the LM head.

        Args:
            pretrained_model_name_or_path:
            roberta_config:
            *args:
            **kwargs:
        """
        super().__init__()

        if pretrained_model_name_or_path is None and roberta_config is None:
            raise ConfigurationError(f"Must provide either a pretrained model name or a Roberta config.")

        config = RobertaConfig(
            **roberta_config, vocab_size=len(tokenizer.get_vocab()), position_embedding_type="relative_key_query"
        )
        if pretrained_model_name_or_path is not None:
            logger.info(f"Initializing transformer stack from a pretrained model {pretrained_model_name_or_path}")
            self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)
        else:
            logger.info(f"Initializing a new Roberta model with config {config}")
            self.encoder = RobertaModel(config=config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config=config)
        self.pad_id = tokenizer.pad_token_id
        self.sgcl_config = sgcl_config

    def _mlm_loss(self, preds, labels):
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        masked_lm_loss = loss_fct(preds.view(-1, self.encoder.config.vocab_size), labels.view(-1))
        return masked_lm_loss

    def forward(self, input_ids, token_type_ids, attention_mask, token_spans, head, deprel, labels=None):
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
        hidden_states = outputs.hidden_states[1:]
        x = outputs.last_hidden_state
        mlm_preds = self.lm_head(x)
        if self.training and labels is not None:
            mlm_loss = self._mlm_loss(mlm_preds, labels)
            sg_loss = syntax_tree_guided_loss(self.sgcl_config, hidden_states, head, token_spans)
            return {"loss": mlm_loss + sg_loss}
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
