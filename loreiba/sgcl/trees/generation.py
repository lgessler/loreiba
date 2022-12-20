import random
from functools import cache
from typing import Any, Dict, List, Set

import torch

from loreiba.sgcl.generation_common import adjacent_ids_of_subtree, get_all_subtrees, get_head_map
from loreiba.sgcl.trees.common import TreeSgclConfig


def get_eligible_subtrees(
    config: TreeSgclConfig, head_map: Dict[int, int | None], all_subtrees: Dict[int, Dict[int, int | None]]
) -> List[Dict[str, Any]]:
    """
    return all subtrees eligible for the tree-based contrastive loss.
    """
    eligible = []

    for token_id in range(1, len(head_map.items())):
        subtree = all_subtrees[token_id]
        # IDs that are not in the subtree but neighbor at least one token in the subtree
        adjacent_ids = adjacent_ids_of_subtree(head_map, set(subtree.keys()))
        # Node count limit for subtree: should be, by default, in the interval [2, 10].
        if not (config.min_subtree_size <= len(subtree) <= config.max_subtree_size):
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


def generate_negative_trees(
    config: TreeSgclConfig,
    all_subtrees: Dict[int, Dict[int, int | None]],
    root_id: int,
    subtree: Dict[int, int | None],
    adjacent_ids: Set[int],
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
    sample_size = min(len(all_replacement_targets), len(all_replacement_values), config.max_replacements)

    already_done = set()
    negatives = []
    retry_count = 0
    while len(negatives) < config.max_negative_per_subtree:
        random_sample_size = random.randint(1, sample_size)
        targets = tuple(sorted(random.sample(all_replacement_targets, random_sample_size)))
        values = tuple(sorted(random.sample(all_replacement_values, random_sample_size)))

        # We've sampled something we already saw. Stop trying if we've exceeded the limit, else try again.
        if (targets, values) in already_done:
            retry_count += 1
            if retry_count > config.max_retry:
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
        # Check the difference in sizes and reject if it's beyond our tolerance (defaults to 1)
        if abs(len(negative_tree) - len(subtree)) > config.max_node_count_difference:
            retry_count += 1
            continue
        negatives.append(negative_tree)

    return {
        "root_id": root_id,
        "positive": subtree,
        "negatives": negatives,
    }


# TODO: is this ok?
@cache
def generate_subtrees(config: TreeSgclConfig, head: torch.LongTensor) -> List[List[Dict[str, Any]]]:
    """
    Generate pairs of positive and negative trees
    """
    # Map from IDs to heads. Note that this is all 1-indexed, with 0 being the dummy ROOT node.
    head_map = get_head_map(head)

    # get eligible subtrees for each sequence
    all_subtree_lists = [get_all_subtrees(s) for s in head_map]
    eligible_subtree_lists = [
        get_eligible_subtrees(config, s, all_subtrees) for s, all_subtrees in zip(head_map, all_subtree_lists)
    ]
    subtree_lists = [config.subtree_sampling_method.sample(subtree_list) for subtree_list in eligible_subtree_lists]
    tree_sets = []
    # For each sentence in the batch...
    for subtree_list, all_subtrees in zip(subtree_lists, all_subtree_lists):
        positive_and_negative_trees = []
        # For each subtree in the sentence...
        for subtree in subtree_list:
            # Collect negative trees with the positive tree
            result = generate_negative_trees(config, all_subtrees, **subtree)
            if len(result["negatives"]) > 1:
                positive_and_negative_trees.append(result)
        tree_sets.append(positive_and_negative_trees)
    return tree_sets
