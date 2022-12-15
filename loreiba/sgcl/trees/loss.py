from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import torch

import loreiba.common as lc
from loreiba.sgcl.trees.common import TreeSgclConfig
from loreiba.sgcl.trees.generation import generate_subtrees
from loreiba.sgcl.trees.info_nce import InfoNCE


def get_root_z(tokenwise_hidden_states, root_id, batch_index):
    num_layers = len(tokenwise_hidden_states)
    root_z = torch.stack(
        [tokenwise_hidden_states[layer_num][batch_index, root_id] for layer_num in range(num_layers)], dim=0
    )
    return root_z


def get_single_tree_vecs(tree, batch_index, root_z, tokenwise_hidden_states):
    """
    Calculate sim_tree according to equation 3
    in section 3.1 of https://aclanthology.org/2022.findings-acl.191.pdf
    """
    num_layers = len(tokenwise_hidden_states)
    # tensor of [num_layers, hidden_size]
    child_ids = [k for k, v in tree.items() if v is not None]
    # tensor of [num_children, num_layers, hidden_size]
    child_zs = torch.stack(
        [
            torch.stack(
                [tokenwise_hidden_states[layer_num][batch_index, child_id] for layer_num in range(num_layers)], dim=0
            )
            for child_id in child_ids
        ],
        dim=0,
    )
    # calculate exponentiated dot products between root and children
    # shape is [num_layers, num_children]
    dots = torch.stack([torch.tensordot(root_z[i], child_zs[:, i, :], dims=([0], [1])) for i in range(num_layers)])
    proportions = dots.softmax(dim=1)
    # this is the right hand argument of cosine in sim_tree in equation 3
    subtree_z = (proportions.swapaxes(1, 0).unsqueeze(-1) * child_zs).sum(0)
    return subtree_z


def assess_tree_sgcl(
    config: TreeSgclConfig,
    tree_sets_for_batch: List[List[Dict[str, Any]]],
    hidden_states: List[torch.Tensor],
    token_spans: torch.LongTensor,
) -> float:
    loss = 0.0
    info_nce = InfoNCE(
        temperature=0.1, reduction="mean", negative_mode="paired", top_k=config.max_negatives_used_in_loss
    )
    tokenwise_hidden_states = [lc.pool_embeddings(layer_i, token_spans) for layer_i in hidden_states]

    # Iterate over items in the batch
    for i, tree_sets in enumerate(tree_sets_for_batch):
        for tree_set in tree_sets:
            # This is 1-indexed, BUT, this actually doesn't need modification because the [CLS] token has shifted
            # everything rightward
            root_id = tree_set["root_id"]
            positive = tree_set["positive"]
            negatives = tree_set["negatives"]
            root_z = get_root_z(tokenwise_hidden_states, root_id, i)
            positive_sims = get_single_tree_vecs(positive, i, root_z, tokenwise_hidden_states)
            negative_sims_list = torch.stack(
                [get_single_tree_vecs(negative, i, root_z, tokenwise_hidden_states) for negative in negatives], dim=1
            )
            nce_term = info_nce(root_z, positive_sims, negative_sims_list)
            loss += nce_term
    return loss / len(tree_sets_for_batch)


if __name__ == None:
    config = lc.dill_load("/tmp/config")
    hidden_states = lc.dill_load("/tmp/hidden_states")
    token_spans = lc.dill_load("/tmp/token_spans")
    head = lc.dill_load("/tmp/head")
    # head_map = {0: None, **{i + 1: h.item() for i, h in enumerate(head[0])}}
    # all_subtrees = get_all_subtrees(config, head_map)
    # eligible_subtrees = sorted(get_eligible_subtrees(config, head_map, all_subtrees), key=lambda x: x["root_id"])
    # output = generate_negative_trees(config, all_subtrees, **eligible_subtrees[2])

    tree_sets_for_batch = generate_subtrees(config, head)
    assess_tree_sgcl(config, tree_sets_for_batch, hidden_states, token_spans)


################################################################################
# top level function
################################################################################
def syntax_tree_guided_loss(
    config: TreeSgclConfig,
    hidden_states: List[torch.Tensor],
    head: torch.LongTensor,
    token_spans: torch.LongTensor,
) -> float:
    """
    Compute the tree-guided contrastive loss presented in Zhang et al. 2022
    (https://aclanthology.org/2022.findings-acl.191.pdf).
    Args:
        config:
            TreeSgclConfig
        hidden_states:
            Has n tensors, where n is the number of layers in the Transformer model.
            Each tensor has shape [batch_size, wordpiece_len, hidden_dim].
        attentions:
            Has n tensors, where n is the number of layers in the Transformer model.
            Each tensor has shape [batch_size, n_heads, wordpiece_len, wordpiece_len].
        token_spans:
            A tensor of shape [batch_size, token_len + 2, 2]: each 2-tuple in the last dim represents the
            wordpiece indices (inclusive on both sides) of the wordpiece span that corresponds to an original
            token that was split by the subword tokenizer. We need this in order to pool hidden representations
            for the tree-based loss term. Note that dim 1 has an extra 2 because of the [CLS] and [SEP] tokens
            that are not included in the syntax tree.
        head:
            A tensor of shape [batch_size, token_len] with indexes of each token's head. Note that these are
            loaded directly from the conllu file, so they are 1-indexed and do not account for special tokens.
            Also note that we do NOT include a sentinel token for ROOT. Consider the following example:
                1   Almaañ      0   root
                2   ci          3   case
                3   jamonoy     1   nmod
                4   Napoleon    3   nmod
            In this case, its entry in `head` would be [0, 3, 1, 3]. Note however that Napoleon's head is NOT at
            index 3. It is instead at index 2, since the tensor is 0-indexed.
    Returns: float
    """
    # print(config)
    # print(token_spans.shape)
    # print(head.shape)
    # print(head[0])
    # print(config)
    # lc.dill_dump(config, "/tmp/config")
    # lc.dill_dump(hidden_states, "/tmp/hidden_states")
    # lc.dill_dump(token_spans, "/tmp/token_spans")
    # lc.dill_dump(head, "/tmp/head")
    tree_sets_for_batch = generate_subtrees(config, head)
    return assess_tree_sgcl(config, tree_sets_for_batch, hidden_states, token_spans)


# subtree(heads, 8)
