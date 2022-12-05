import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tango.common.exceptions import ConfigurationError
from tango.integrations.torch import Model
from tango.integrations.transformers import Config
from transformers import AutoModel, RobertaConfig, RobertaModel

logger = logging.getLogger(__name__)


@Model.register("loreiba.sgcl.model::sgcl_model")
class SGCLModel(Model):
    """
    Re-implementation of Syntax-Guided Contrastive Loss for Pre-trained Language Model
    (https://aclanthology.org/2022.findings-acl.191.pdf).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        roberta_config: Dict[str, Any] = None,
        *args,
        **kwargs,
    ):
        """
        Provide EITHER `pretrained_model_name_or_path` OR `roberta_config`.

        Args:
            pretrained_model_name_or_path:
            roberta_config:
            *args:
            **kwargs:
        """
        super().__init__()
        self.x = nn.Parameter(torch.tensor([1.0]))

        if pretrained_model_name_or_path is None and roberta_config is None:
            raise ConfigurationError(f"Must provide either a pretrained model name or a Roberta config.")
        if pretrained_model_name_or_path is not None:
            logger.info(f"Initializing transformer stack from a pretrained model {pretrained_model_name_or_path}")
            self.transformer_stack = AutoModel.from_pretrained(pretrained_model_name_or_path)
        else:
            config = RobertaConfig(**roberta_config)
            logger.info(f"Initializing a new Roberta model with config {config}")
            self.transformer_stack = RobertaModel(config=config)

    def forward(self, input_ids, token_type_ids, attention_mask, token_indexes, head, deprel):
        assert False
        pass
