"""
Implementation of tree generation and contrastive loss based on Zhang et al. 2022.
Note the following additional implementation details that were obtained in correspondence
with Shuai Zhang:

    1. What is the minimum number of nodes for a subtree that is being used to generate negative trees?
       Based on other constraints (a generated negative tree must have exactly 1, 2, or 3 nodes changed,
       and there must be at least one token in common between the original (positive) and negative tree),
       I think the absolute smallest subtree would be one with two nodes: one node would then stay the same,
       and the other would be replaced to generate a negative tree. So is 2 nodes the actual minimum, or was it higher?
    2. The paper says that "no more than three tokens" are replaced when generating a negative subtree.
       Does this mean that you try to replace 3 whenever possible, and replace 2 or 1 only when there are only 2 or 1
       possible substitute tokens?
    3. For each sentence, were all tokens in the sentence (subject to constraints) used to generate negative subtrees?
       Or were only a few tokens selected, perhaps randomly?
    4. For a given subtree, how many negative subtrees are generated?
    5. Are tokens that are replaced in order to make a negative tree leaf nodes in the subtree,
       or can nodes that are replaced be non-leafs?

Answers:

    1. The minimum and maximum number of nodes for a subtree are 2 and 10, respectively.
    2. In the implementation process, we did not strictly follow the rule of "no more than three tokens",
       which is just a generalization. Taking the example in the paper "We donated wire for the convicts to
       build cages with." as an example, for the given subtree "to build cages with" (four tokens), we randomly
       add the left (right) adjacent token to the subtree and delete the token on the right (left) side of the subtree,
       generating the following examples:
        a) convicts to build cages with
        b) convicts to build cages
        c) convicts to build
        d) the convicts to build cages
        e) the convicts to build
        f) build cages with .
       We mainly follow the following two principles when generating negative subtrees:
        a) The generated negative subtree must contain the root node of the subtree ("build").
        b) The absolute difference between the number of tokens in the generated negative subtree
           and the given subtree does not exceed one.
    3. All tokens that are non-leaf nodes of the tree were used to generate subtrees.
    4. There was generated at most 30 negative subtrees, but only the top three in terms of simtree were used
       in the calculation of the loss function.
    5. The answer to this question can be found in the answer to question 2.
"""
import random
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import dill
import torch
import torch.nn.functional as F
from info_nce import InfoNCE
from tango.common import FromParams, Registrable

import loreiba.common as lc


################################################################################
# Subtree sampling methods
################################################################################
class SubtreeSamplingMethod(Registrable):
    """Subclasses of this specify how to select subtrees in a given sentence."""

    def sample(self, subtrees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplemented()


@SubtreeSamplingMethod.register("random")
class RandomSubtreeSamplingMethod(SubtreeSamplingMethod):
    """Sample subtrees by randomly selecting up to `max_number` subtrees in the sentence."""

    def __init__(self, max_number: int):
        self.max_number = max_number

    def sample(self, subtrees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return random.sample(subtrees, min(self.max_number, len(subtrees)))


@SubtreeSamplingMethod.register("all")
class AllSubtreeSamplingMethod(SubtreeSamplingMethod):
    """Sample subtrees by using all subtrees."""

    def sample(self, subtrees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return subtrees


################################################################################
# Config
################################################################################
class TreeSgclConfig(FromParams):
    def __init__(
        self,
        max_negative_per_subtree: int = 3,
        subtree_sampling_method: SubtreeSamplingMethod = AllSubtreeSamplingMethod(),
    ):
        self.max_negative_per_subtree = max_negative_per_subtree
        self.subtree_sampling_method = subtree_sampling_method


################################################################################
# Code
################################################################################
def immediate_children(head_map: Dict[int, int | None], token_id: int):
    return [child for child, parent in head_map.items() if parent == token_id]


def subtree_of_id(head_map: Dict[int, int | None], token_id: int) -> Dict[int, int | None]:
    """Get the subtree rooted at a given ID."""
    subtree = {token_id: None}
    queue = immediate_children(head_map, token_id)
    while len(queue) > 0:
        token_id = queue.pop(0)
        subtree[token_id] = head_map[token_id]
        children = immediate_children(head_map, token_id)
        queue.extend(children)
    return subtree


def adjacent_ids_of_subtree(head_map: Dict[int, int | None], subtree_ids: Set[int]) -> Set[int]:
    """
    Return set of IDs that are adjacent to all IDs in a subtree
    and are NOT a direct ancestor of the subtree's root
    """
    adjacent = set()

    # Get parents
    parent_ids = set()
    current = tuple(subtree_ids)[0]
    while current is not None:
        parent_ids.add(current)
        current = head_map[current]

    # On to the main work
    for token_id in subtree_ids:
        left = token_id - 1
        right = token_id + 1
        for x in [left, right]:
            if x > 0 and x not in subtree_ids and x not in parent_ids and x in head_map.keys():
                adjacent.add(x)
    return adjacent


def get_all_subtrees(head_map: Dict[int, int | None]) -> Dict[int, Dict[int, int | None]]:
    all = {}
    for token_id in range(1, len(head_map.items())):
        all[token_id] = subtree_of_id(head_map, token_id)
    return all


def get_eligible_subtrees(
    head_map: Dict[int, int | None], all_subtrees: Dict[int, Dict[int, int | None]]
) -> List[Dict[str, Any]]:
    """
    Given a tensor of shape [token_len], return all subtrees and subtrees eligible for the tree-based contrastive loss.
    """
    eligible = []

    for token_id in range(1, len(head_map.items())):
        subtree = all_subtrees[token_id]
        # IDs that are not in the subtree but neighbor at least one token in the subtree
        adjacent_ids = adjacent_ids_of_subtree(head_map, set(subtree.keys()))
        # We need at least one token to replace and one token to stay the same
        if len(subtree) < 2:
            continue
        # We need at least one token to provide as a replacement
        if len(adjacent_ids) < 1:
            continue
        eligible.append(
            {
                "root_id": token_id,
                "subtree": subtree,
                "adjacent_ids": adjacent_ids,
            }
        )
    return eligible


################################################################################
# tree generation
################################################################################
def generate_negative_trees(
    config: TreeSgclConfig,
    all_subtrees: Dict[int, Dict[int, int | None]],
    root_id: int,
    subtree: Dict[int, int | None],
    adjacent_ids: Set[int],
    max_retry: int = 10,
    max_replacements: int = 3,
) -> Dict[str, Any]:
    """
    There are potentially many subtrees, but we only want up to `config.max_negative_per_subtree`.
    Define a negative tree in terms of:
      (1) a list of exactly one, two, or three nodes
      (2) a list of adjacent nodes, equal in size to (1), which are the replacements for the leaf nodes in (1)
    We could try to generate all possible negative combinations of the two and sample from that, but
    for efficiency we're just going to sample and check for duplicates, breaking out of generation if we
    get a collision `max_retry` times in a row.
    """
    # A valid node for replacement is any one that is not the root of the subtree
    all_replacement_targets = tuple({k for k, v in subtree.items() if v is not None})
    # A valid replacement is an adjacent id
    all_replacement_values = tuple(adjacent_ids.copy())
    # 3 possible limiting reagents for replacements: targets, values, and the limit (a hyperparameter)
    sample_size = min(len(all_replacement_targets), len(all_replacement_values), max_replacements)

    already_done = set()
    negatives = []
    retry_count = 0
    while len(negatives) < config.max_negative_per_subtree:
        targets = tuple(sorted(random.sample(all_replacement_targets, sample_size)))
        values = tuple(sorted(random.sample(all_replacement_values, sample_size)))

        # We've sampled something we already saw. Stop trying if we've exceeded the limit, else try again.
        if (targets, values) in already_done:
            retry_count += 1
            if retry_count > max_retry:
                break
            else:
                continue
        # This is new--record that we saw it
        already_done.add((targets, values))

        # First, copy the subtree
        negative_tree = subtree.copy()
        for target, value in zip(targets, values):
            # Retrieve the subtree to be removed
            target_subtree = all_subtrees[target]
            # Note the head of the subtree we're deleting
            subtree_head_id = [k for k, v in target_subtree.items() if v is None][0]
            # Remove the target: find all the token IDs in the subtree and remove them
            for k in target_subtree.keys():
                # We might have already removed the subtree in a previous iteration, so check first
                if k in negative_tree:
                    del negative_tree[k]
            # Retrieve the subtree to be spliced into the negative tree
            replacement_subtree = all_subtrees[value]
            for token_id, head_id in replacement_subtree.items():
                # if we found the root of the replacement subtree, make its head the original subtree's head
                if head_id is None:
                    head_id = subtree_head_id
                negative_tree[token_id] = head_id
        negatives.append(negative_tree)

    return {
        "root_id": root_id,
        "positive": subtree,
        "negatives": negatives,
    }


def generate_subtrees(config: TreeSgclConfig, head: torch.LongTensor) -> List[List[Dict[str, Any]]]:
    """
    Generate pairs of positive and negative trees
    """
    # Count number of tokens in the tree: find nonzero heads to account for 0-padding, and add one
    # to account for the fact that 0:root is labeled with a head = 0.
    token_counts = (head != 0).sum(1) + 1

    # split the batched head tensor into one tensor per input sequence, with padding removed
    padless_head = [head[i, : token_counts[i]] for i, x in enumerate(token_counts)]
    # Map from IDs to heads. Note that this is all 1-indexed, with 0 being the dummy ROOT node.
    head_map = [{0: None, **{i + 1: h.item() for i, h in enumerate(heads)}} for heads in padless_head]

    # get eligible subtrees for each sequence
    all_subtree_lists = [get_all_subtrees(s) for s in head_map]
    eligible_subtree_lists = [
        get_eligible_subtrees(s, all_subtrees) for s, all_subtrees in zip(head_map, all_subtree_lists)
    ]
    subtree_lists = [config.subtree_sampling_method.sample(subtree_list) for subtree_list in eligible_subtree_lists]
    tree_sets = []
    # For each sentence in the batch...
    for subtree_list, all_subtrees in zip(subtree_lists, all_subtree_lists):
        positive_and_negative_trees = []
        # For each subtree in the sentence...
        for subtree in subtree_list:
            # Collect negative trees with the positive tree
            positive_and_negative_trees.append(generate_negative_trees(config, all_subtrees, **subtree))
        tree_sets.append(positive_and_negative_trees)
    return tree_sets


################################################################################
# loss calculation
################################################################################
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
    info_nce = InfoNCE(temperature=0.1, reduction="mean", negative_mode="paired")
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
    attentions = lc.dill_load("/tmp/attentions")
    hidden_states = lc.dill_load("/tmp/hidden_states")
    token_spans = lc.dill_load("/tmp/token_spans")
    head = lc.dill_load("/tmp/head")
    # head_map = {0: None, **{i + 1: h.item() for i, h in enumerate(head[0])}}
    # all_subtrees = get_all_subtrees(config, head_map)
    # eligible_subtrees = sorted(get_eligible_subtrees(config, head_map, all_subtrees), key=lambda x: x["root_id"])
    # output = generate_negative_trees(config, all_subtrees, **eligible_subtrees[2])

    tree_sets_for_batch = generate_subtrees(config, head)
    tree_sets_for_batch[0]
    assess_tree_sgcl(config, tree_sets_for_batch, hidden_states, attentions, token_spans)
    tokenwise_hidden_states[0].shape

    info_nce = InfoNCE(temperature=0.1, reduction="mean", negative_mode="paired")


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
    # lc.dill_dump(attentions, "/tmp/attentions")
    # lc.dill_dump(hidden_states, "/tmp/hidden_states")
    # lc.dill_dump(token_spans, "/tmp/token_spans")
    # lc.dill_dump(head, "/tmp/head")
    tree_sets_for_batch = generate_subtrees(config, head)
    return assess_tree_sgcl(config, tree_sets_for_batch, hidden_states, token_spans)


# subtree(heads, 8)
