import torch
import torch.nn as nn
from tango.integrations.torch import Model


@Model.register("loreiba.sgcl.model::sgcl_model")
class SGCLModel(Model):
    """
    Re-implementation of Syntax-Guided Contrastive Loss for Pre-trained Language Model
    (https://aclanthology.org/2022.findings-acl.191.pdf).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.x = nn.Parameter(torch.tensor([1.0]))

    def forward(self, input_ids, token_type_ids, attention_mask, token_indexes, head, deprel):
        assert False
        pass
