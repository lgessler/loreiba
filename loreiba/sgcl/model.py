import logging
import os
from typing import Any, Dict, Optional

import psutil
import torch
import torch.nn.functional as F
from _socket import gethostname
from tango.common import Registrable
from tango.common.exceptions import ConfigurationError
from tango.integrations.torch import Model, TrainCallback
from tango.integrations.transformers import Tokenizer
from transformers import AutoModel, BertConfig, BertModel, ElectraConfig, ElectraModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.electra.modeling_electra import ElectraDiscriminatorPredictions
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from loreiba.common import dill_dump, dill_load
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
        self.head = RobertaLMHead(config=config)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def compute_loss(self, input_ids, attention_mask, token_type_ids, last_encoder_state, labels):
        preds = self.head(last_encoder_state)
        if not (labels != -100).any():
            return 0.0
        masked_lm_loss = F.cross_entropy(preds.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)
        return {"mlm": masked_lm_loss}


@SgclEncoder.register("electra")
class ElectraEncoder(SgclEncoder):
    """
    We will follow the simplest approach described in the paper (https://openreview.net/pdf?id=r1xMH1BtvB),
    which is to tie all the weights of the discriminator and the generator. In effect, this means we can
    just use the same Transformer encoder stack for both the discriminator and the generator, with different
    heads on top for MLM and replaced token detection.
    """

    def __init__(self, tokenizer: Tokenizer, electra_config: Dict[str, Any]):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id
        config = ElectraConfig(
            **electra_config, vocab_size=len(tokenizer.get_vocab()), position_embedding_type="relative_key_query"
        )
        logger.info(f"Initializing a new BERT model with config {config}")
        self.config = config
        self.encoder = ElectraModel(config=config)
        self.tokenizer = tokenizer

        # Just use the Roberta head
        self.generator_head = RobertaLMHead(config=config)
        self.discriminator_head = ElectraDiscriminatorPredictions(config=config)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def compute_loss(self, input_ids, attention_mask, token_type_ids, last_hidden_state, labels):
        """
        Used to compute discriminator's loss
        """
        mlm_logits = self.generator_head(last_hidden_state)
        if not (labels != -100).any():
            masked_lm_loss = 0.0
        else:
            masked_lm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100
            )

        # Take predicted token IDs. Note that argmax() breaks the gradient chain, so the generator only learns from MLM
        mlm_preds = mlm_logits.argmax(-1)
        # Combine them to get labels for discriminator
        replaced = (~input_ids.eq(mlm_preds)) & (labels != -100)

        # Make inputs for discriminator, feed them into the encoder once more, then feed to discriminator head
        replaced_input_ids = torch.where(replaced, mlm_preds, input_ids)
        second_encoder_output = self.encoder(
            input_ids=replaced_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        discriminator_output = self.discriminator_head(second_encoder_output.last_hidden_state)

        # compute replaced token detection BCE loss on eligible tokens (all non-special tokens)
        bce_mask = (
            (replaced_input_ids != self.tokenizer.cls_token_id)
            & (replaced_input_ids != self.tokenizer.sep_token_id)
            & (replaced_input_ids != self.tokenizer.pad_token_id)
        )
        rtd_preds = torch.masked_select(discriminator_output, bce_mask)
        rtd_labels = torch.masked_select(replaced.float(), bce_mask)
        rtd_loss = F.binary_cross_entropy_with_logits(rtd_preds, rtd_labels)

        # dill_dump(input_ids, '/tmp/input_ids')
        # dill_dump(attention_mask, '/tmp/attention_mask')
        # dill_dump(token_type_ids, '/tmp/token_type_ids')
        # dill_dump(last_hidden_state, '/tmp/last_hidden_state')
        # dill_dump(labels, '/tmp/labels')
        # dill_dump(self, '/tmp/self')
        return {"rtd": (100 * rtd_loss), "mlm": masked_lm_loss}


def tmp():
    input_ids = dill_load("/tmp/input_ids")[:32]
    attention_mask = dill_load("/tmp/attention_mask")[:32]
    token_type_ids = dill_load("/tmp/token_type_ids")[:32]
    last_hidden_state = dill_load("/tmp/last_hidden_state")[:32]
    labels = dill_load("/tmp/labels")[:32]
    self = dill_load("/tmp/self")


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
