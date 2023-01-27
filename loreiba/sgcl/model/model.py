import logging
import os
from copy import copy
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import torch
import torch.nn.functional as F
from _socket import gethostname
from tango.common import Registrable
from tango.common.exceptions import ConfigurationError
from tango.integrations.torch import Model, TrainCallback
from tango.integrations.transformers import Tokenizer
from torch import nn
from transformers import AutoModel, BertConfig, BertModel, ElectraConfig, ElectraModel
from transformers.activations import GELUActivation, gelu, get_activation
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.electra.modeling_electra import ElectraDiscriminatorPredictions, ElectraGeneratorPredictions
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from loreiba.common import dill_dump, dill_load
from loreiba.sgcl.model.encoder import SgclEncoder
from loreiba.sgcl.phrases.common import PhraseSgclConfig
from loreiba.sgcl.phrases.loss import phrase_guided_loss
from loreiba.sgcl.trees.common import TreeSgclConfig
from loreiba.sgcl.trees.loss import syntax_tree_guided_loss

logger = logging.getLogger(__name__)


################################################################################
# Main model
################################################################################
@Model.register("loreiba.sgcl.model.model::sgcl_model")
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

        self.tree_sgcl_config = tree_sgcl_config
        self.phrase_sgcl_config = phrase_sgcl_config

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        token_spans,
        xpos,
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

        if labels is not None:
            outputs = {}
            head_loss = self.encoder.compute_loss(input_ids, attention_mask, token_type_ids, last_encoder_state, labels)
            loss = sum(head_loss.values())

            outputs["mlm_loss"] = head_loss["mlm"]
            outputs["progress_items"] = {
                "max_cuda_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "resident_memory_mb": psutil.Process().memory_info().rss / 1024**2,
                "mlm_loss": head_loss["mlm"].item(),
                "perplexity": head_loss["mlm"].exp().item(),
            }
            if "rtd" in head_loss:
                outputs["progress_items"]["rtd_loss"] = head_loss["rtd"].item()

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
    def __init__(self, path: str, model_attr: Optional[str] = None, use_best: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.model_attr = model_attr
        self.use_best = use_best

    def post_train_loop(self, step: int, epoch: int) -> None:
        # Always use the last state path
        state_path = self.train_config.state_path if not self.use_best else self.train_config.best_state_path
        state = torch.load(state_path / Path("worker0_model.pt"), map_location="cpu")
        model = self.model.cpu()
        model.load_state_dict(state, strict=True)

        # Get the target attr
        if self.model_attr:
            for piece in self.model_attr.split("."):
                model = getattr(model, piece)

        # Save in the HuggingFace format
        model.save_pretrained(self.path)
        self.logger.info(f"Wrote model to {self.path}")
