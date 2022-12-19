import random
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from loreiba.common import dill_dump, dill_load
from loreiba.sgcl.generation_common import get_all_subtrees, get_head_map
from loreiba.sgcl.phrases.common import PhraseSgclConfig


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


def depth_of_tree(root: int, t: Dict[int, int | None]) -> int:
    max_depth = 1
    queue = [(k, 1) for k, v in t.items() if v == root]
    while len(queue) > 0:
        current, parent_depth = queue.pop(0)
        depth = parent_depth + 1
        if depth > max_depth:
            max_depth = depth
        children = [(k, depth) for k, v in t.items() if v == current]
        queue.extend(children)
    return max_depth


def compute_phrase_loss(
    config: PhraseSgclConfig,
    batch_index: int,
    head_map: Dict[int, int | None],
    all_subtrees: Dict[int, Dict[int, int | None]],
    averaged_attentions: torch.Tensor,
    t2wp: Dict[int, int],
    temperature: float = 0.1,
) -> List[float]:
    losses = []
    jsd = JSD()
    wordpiece_length = max(t2wp.values()) + 1

    for root, subtree in all_subtrees.items():
        depth = depth_of_tree(root, subtree)
        if depth > config.max_subtree_height:
            continue
        if len(subtree.keys()) < config.min_subtree_token_count:
            continue
        tokens_not_in_phrase = [token_id for token_id in head_map.keys() if token_id not in subtree and token_id > 0]
        if len(tokens_not_in_phrase) == 0:
            continue

        positive = t2wp[random.sample(tuple(subtree.keys()), 1)[0]]
        query = t2wp[random.sample(tuple(set(subtree.keys()) - {positive}), 1)[0]]
        negatives = [
            t2wp[i]
            for i in random.sample(tokens_not_in_phrase, min(config.negative_per_positive, len(tokens_not_in_phrase)))
        ]

        query_reprs = averaged_attentions[:, batch_index, query, :wordpiece_length]
        positive_reprs = averaged_attentions[:, batch_index, positive, :wordpiece_length]
        negative_reprs = torch.stack(
            [averaged_attentions[:, batch_index, negative, :wordpiece_length] for negative in negatives], dim=0
        )

        positive_sim = torch.stack([jsd(qr, pr) for qr, pr in zip(query_reprs, positive_reprs)])
        negative_sims = []
        for n in negative_reprs:
            negative_sims.append(torch.stack([jsd(qr, nr) for qr, nr in zip(query_reprs, n)]))
        negative_sims = torch.stack(negative_sims, dim=1)

        combined_sims = torch.hstack((positive_sim.reshape(-1, 1), negative_sims)) / temperature
        # take the softmax for InfoNCE
        softmaxed = -F.log_softmax(combined_sims, dim=1)
        #
        loss = softmaxed[:, 0]
        losses.append(loss.mean().item())

    # one idea:
    # - preprocess by generating token IDs
    # - batch everything together in a [batch_count, max_num_subtrees, max_num_neg]
    # - use these indices to get the right attentions and batch as much as possible

    return sum(losses) / len(losses) if len(losses) > 0 else 0.0


def get_token_to_head_wordpiece_map(spans):
    m = {}
    for i, span in enumerate(spans):
        k = span[0].item()
        if k == -1:
            break
        if i not in m:
            m[i] = k
    return m


def phrase_guided_loss(
    config: PhraseSgclConfig, attentions: List[torch.FloatTensor], token_spans: torch.Tensor, head: torch.LongTensor
) -> float:
    attentions = torch.stack(attentions, dim=0)
    averaged_attentions = attentions.mean(dim=2)
    head_map = get_head_map(head)
    token_to_head_wordpiece_maps = [get_token_to_head_wordpiece_map(spans) for spans in token_spans]
    subtrees = [get_all_subtrees(s) for s in head_map]

    losses = [
        compute_phrase_loss(config, i, s, all_subtrees, averaged_attentions, t2wp)
        for i, (s, all_subtrees, t2wp) in enumerate(zip(head_map, subtrees, token_to_head_wordpiece_maps))
    ]

    # dill_dump(config, "/tmp/config")
    # dill_dump(attentions, "/tmp/attentions")
    # dill_dump(token_spans, "/tmp/token_spans")
    # dill_dump(head, "/tmp/head")
    return sum(losses) / len(losses) if len(losses) > 0 else 0.0


def tmp():
    config = dill_load("/tmp/config")
    attentions = dill_load("/tmp/attentions")
    head = dill_load("/tmp/head")
    token_spans = dill_load("/tmp/token_spans")
    attentions = torch.stack(attentions, dim=0)
