from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import torch
import torch.nn.functional as F

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

    # calculate dot products between root and children and softmax them
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
    tokenwise_hidden_states = torch.stack([lc.pool_embeddings(layer_i, token_spans) for layer_i in hidden_states])

    # Iterate over items in the batch
    print([len(s) for s in tree_sets_for_batch])
    for i, tree_sets in enumerate(tree_sets_for_batch):
        for tree_set in tree_sets:
            # This is 1-indexed, BUT, this actually doesn't need modification because the [CLS] token has shifted
            # everything rightward
            root_id = tree_set["root_id"]
            positive = tree_set["positive"]
            negatives = tree_set["negatives"]
            root_z = get_root_z(tokenwise_hidden_states, root_id, i)
            positive_vecs = get_single_tree_vecs(positive, i, root_z, tokenwise_hidden_states)
            negative_vecs_list = torch.stack(
                [get_single_tree_vecs(negative, i, root_z, tokenwise_hidden_states) for negative in negatives], dim=1
            )
            nce_term = info_nce(root_z, positive_vecs, negative_vecs_list)
            loss += nce_term
    return loss / len(tree_sets_for_batch)


def assess_tree_sgcl_batched(
    config: TreeSgclConfig,
    tree_sets_for_batch: List[List[Dict[str, Any]]],
    hidden_states: List[torch.Tensor],
    token_spans: torch.LongTensor,
    temperature: float = 0.1,
) -> float:
    device = hidden_states[0].device
    tokenwise_hidden_states = torch.stack([lc.pool_embeddings(layer_i, token_spans) for layer_i in hidden_states])

    num_layers, batch_size, sequence_length, hidden_dim = tokenwise_hidden_states.shape
    root_ids = defaultdict(list)
    positives = defaultdict(list)
    negative_lists = defaultdict(list)
    total_subtrees = 0
    max_positive_ids = 0
    max_negatives = 0
    max_negative_ids = 0
    for i, tree_sets in enumerate(tree_sets_for_batch):
        if len(tree_sets) == 0:
            continue
        all_negative_ids = [
            [[k for k in negative.keys()] for negative in tree_set["negatives"]] for tree_set in tree_sets
        ]
        max_positive_length = max(len(tree_set["positive"].keys()) for tree_set in tree_sets)
        max_negative_length = max(max([len(negative) for negative in negatives]) for negatives in all_negative_ids)
        max_negative_count = max(len(negatives) for negatives in all_negative_ids)
        for tree_set in tree_sets:
            root_id = tree_set["root_id"]
            root_ids[i].append(root_id)
            positive_ids = [root_id] + [k for k in tree_set["positive"].keys() if k != root_id]
            positive_ids = positive_ids + ([-1] * (max_positive_length - len(positive_ids)))
            positives[i].append(positive_ids)
            negative_ids = [
                [root_id] + [k for k in negative.keys() if k != root_id] for negative in tree_set["negatives"]
            ]
            padded_negative_ids = [
                ([k for k in ids] + ([-1] * (max_negative_length - len(ids)))) for ids in negative_ids
            ]
            while len(padded_negative_ids) < max_negative_count:
                padded_negative_ids.append([-1] * max_negative_length)
            negative_lists[i].append(padded_negative_ids)
            total_subtrees += 1
            if len(negative_ids) > max_negatives:
                max_negatives = len(tree_set["negatives"])
            if max(len(x) for x in negative_ids) > max_negative_ids:
                max_negative_ids = max(len(x) for x in negative_ids)
            if len(positive_ids) > max_positive_ids:
                max_positive_ids = len(positive_ids)
    max_subtrees = max(len(x) for x in positives.values())

    root_indexes = torch.full((batch_size, max_subtrees), -1, dtype=torch.long, device=device)
    positive_indexes = torch.full((batch_size, max_subtrees, max_positive_ids), -1, dtype=torch.long, device=device)
    negative_indexes = torch.full(
        (batch_size, max_subtrees, max_negatives, max_negative_ids), -1, dtype=torch.long, device=device
    )
    i = 0
    for i in range(len(root_ids)):
        n = len(root_ids[i])
        if n == 0:
            continue
        root_indexes[i, :n] = torch.tensor(root_ids[i])
        positives_for_set = torch.tensor(positives[i])
        positive_indexes[i, : positives_for_set.shape[0], : positives_for_set.shape[1]] = positives_for_set
        negatives_for_set = torch.tensor(negative_lists[i])
        a, b, c = negatives_for_set.shape
        negative_indexes[i, :a, :b, :c] = negatives_for_set

    negative_mask = negative_indexes.ne(torch.tensor(-1, dtype=torch.long, device=device))
    positive_mask = positive_indexes.ne(torch.tensor(-1, dtype=torch.long, device=device))
    root_mask = root_indexes.ne(torch.tensor(-1, dtype=torch.long, device=device))

    root_indexes = root_indexes.clamp(min=0)
    positive_indexes = positive_indexes.clamp(min=0)
    negative_indexes = negative_indexes.clamp(min=0)

    # Find root vectors
    num_layers = tokenwise_hidden_states.shape[0]
    num_hidden = tokenwise_hidden_states.shape[-1]
    root_indexes = root_indexes.unsqueeze(0).unsqueeze(-1).repeat(num_layers, 1, 1, num_hidden)
    root_zs = tokenwise_hidden_states.take_along_dim(root_indexes, dim=2) * root_mask.unsqueeze(0).repeat(
        3, 1, 1
    ).unsqueeze(-1)

    # Find positive vectors
    flattened_positive_number = positive_indexes.shape[1] * positive_indexes.shape[2]
    positive_target_shape = (num_layers, batch_size, flattened_positive_number, num_hidden)
    flattened_positive_indexes = (
        positive_indexes.unsqueeze(0).unsqueeze(-1).repeat(num_layers, 1, 1, 1, num_hidden).view(positive_target_shape)
    )
    positive_zs = tokenwise_hidden_states.take_along_dim(flattened_positive_indexes, dim=2)

    # Find negative vectors
    flattened_negative_number = negative_indexes.shape[1] * negative_indexes.shape[2] * negative_indexes.shape[3]
    negative_target_shape = (num_layers, batch_size, flattened_negative_number, num_hidden)
    flattened_negative_indexes = (
        negative_indexes.unsqueeze(0)
        .unsqueeze(-1)
        .repeat(num_layers, 1, 1, 1, 1, num_hidden)
        .view(negative_target_shape)
    )
    negative_zs = tokenwise_hidden_states.take_along_dim(flattened_negative_indexes, dim=2)

    # Now we need to compute dot products
    # shape: [num_layers, batch_size, max_pairs_for_batch_item, max_ids_for_positive_subtree, hidden_dim]
    reshaped_positive_zs = positive_zs.view(num_layers, *positive_indexes.shape, num_hidden)
    # shape: [num_layers, batch_size, max_pairs_for_batch_item, max_num_negative_max_ids_for_positive_subtree, hidden_dim]
    reshaped_negative_zs = negative_zs.view(num_layers, *negative_indexes.shape, num_hidden)

    positive_dots = torch.einsum("abch,abcdh->abcd", root_zs, reshaped_positive_zs)
    positive_dots = positive_dots * positive_mask.unsqueeze(0)
    negative_dots = torch.einsum("abch,abcdeh->abcde", root_zs, reshaped_negative_zs)
    negative_dots = negative_dots * negative_mask.unsqueeze(0)

    positive_eij = F.softmax(positive_dots, dim=-1) * positive_mask.unsqueeze(0)
    positive_sim_rhs = (reshaped_positive_zs * positive_eij.unsqueeze(-1)).sum(-2)

    negative_eij = F.softmax(negative_dots, dim=-1) * negative_mask.unsqueeze(0)
    negative_sim_rhs = (reshaped_negative_zs * negative_eij.unsqueeze(-1)).sum(-2)

    positive_cosines = F.cosine_similarity(root_zs, positive_sim_rhs, dim=-1)
    negative_cosines = F.cosine_similarity(root_zs.unsqueeze(-2), negative_sim_rhs, dim=-1)

    combined = torch.concat((positive_cosines.unsqueeze(-1), negative_cosines), dim=-1)

    losses = -F.log_softmax(combined / temperature, dim=-1)[:, :, :, 0]
    softmax_mask = root_mask.unsqueeze(0).repeat(3, 1, 1)
    losses = softmax_mask * losses
    loss = losses.sum(-1).mean(dim=0).mean(dim=0)

    print()
    print(losses[0, 0:4])
    print(loss)

    return loss


################################################################################
# top level function
################################################################################
def syntax_tree_guided_loss(
    config: TreeSgclConfig,
    hidden_states: List[torch.Tensor],
    token_spans: torch.LongTensor,
    head: torch.LongTensor,
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
    lc.dill_dump(config, "/tmp/config")
    lc.dill_dump(hidden_states, "/tmp/hidden_states")
    lc.dill_dump(token_spans, "/tmp/token_spans")
    lc.dill_dump(head, "/tmp/head")
    tree_sets_for_batch = generate_subtrees(config, head)
    return assess_tree_sgcl_batched(config, tree_sets_for_batch, hidden_states, token_spans)


def scratch():
    config = lc.dill_load("/tmp/config")
    hidden_states = lc.dill_load("/tmp/hidden_states")
    token_spans = lc.dill_load("/tmp/token_spans")
    head = lc.dill_load("/tmp/head")
    # head_map = {0: None, **{i + 1: h.item() for i, h in enumerate(head[0])}}
    # all_subtrees = get_all_subtrees(config, head_map)
    # eligible_subtrees = sorted(get_eligible_subtrees(config, head_map, all_subtrees), key=lambda x: x["root_id"])
    # output = generate_negative_trees(config, all_subtrees, **eligible_subtrees[2])

    tree_sets_for_batch = generate_subtrees(config, head)
    assess_tree_sgcl_batched(config, tree_sets_for_batch, hidden_states, token_spans)
