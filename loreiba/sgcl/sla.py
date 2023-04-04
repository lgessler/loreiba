from collections import defaultdict
from typing import Any, Dict, List

import torch
from tango.common import FromParams

from loreiba.common import dill_dump, dill_load
from loreiba.sgcl.generation_common import get_head_map


class SlaConfig(FromParams):
    def __init__(self, max_distance: int = 3):
        self.max_distance = max_distance


def get_adjacency_map(head_map: Dict[int, int | None]) -> Dict[int, Dict[int, bool]]:
    adj_map = defaultdict(lambda: defaultdict(bool))
    for k, v in head_map.items():
        if k is not None and v is not None:
            adj_map[k][v] = True
            adj_map[v][k] = True
    return adj_map


def ids_in_range(head_map: Dict[int, int | None], begin: int, max_distance: int):
    head_map = {k: v for k, v in head_map.items() if k != 0 and v != 0}
    adj_map = get_adjacency_map(head_map)
    if begin not in adj_map:
        return set()

    visited = set()

    def inner(current: int, distance: int):
        if distance > max_distance:
            return
        visited.add(current)

        neighbors = {k for k, v in adj_map[current].items() if v and k not in visited}
        for v in neighbors:
            inner(v, distance + 1)

    inner(begin, 0)
    return visited


def generate_sla_mask(config: SlaConfig, head: torch.LongTensor) -> torch.LongTensor:
    """
    Given head of [batch_size, seq_len] s.t.:
     - [CLS] and [SEP] are NOT present
     - the root of the sequence represented in head is equal to 0
    produce [batch_size, seq_len + 2, seq_len + 2] (we expand to account for [CLS] and [SEP])
    such that [b, i, j] is 1 if token i may attend to token j, and 0 otherwise. Note that
    whenever j corresponds to [CLS] or [SEP] or when i = j, [b, i, j] is always 1.
    """
    batch_size, max_seq_len = head.shape
    device = head.device
    zero_col = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
    head = torch.concat((zero_col, head, zero_col), dim=1)
    head_maps = get_head_map(head[:, 1:])
    att_mask: torch.LongTensor = torch.zeros(head.shape + (head.shape[1],), dtype=torch.long, device=head.device)
    for b, head_map in enumerate(head_maps):
        for i in range(1, head.shape[1] - 1):
            in_range = ids_in_range(head_map, i, config.max_distance)
            for j in in_range:
                att_mask[b, i, j] = 1
    # Ensure [CLS] is not masked
    att_mask[:, :, 0] = 1
    non_packed_seq_lens = (head != 0).sum(1) + 2
    for i, l in enumerate(non_packed_seq_lens):
        # Allow [CLS] to attend to all tokens. (This is a point of departure vs. the original impl.)
        att_mask[i, 0, :l] = 1
        # Ensure [SEP] is not masked
        att_mask[i, :, l] = 1
        # Mask out rows beyond sequence length
        att_mask[i, l + 1 :] = 0
    return att_mask


def __tmp():
    head = dill_load("/tmp/head")
    config = dill_load("/tmp/config")
    head = torch.tensor(
        [
            [5, 0, 2, 2, 4],
            [0, 3, 4, 1, 0],
        ],
        dtype=torch.long,
    )

    sla_mask = generate_sla_mask(config, head)
    sla_mask[1, 0].sum()
    (head[1] != 0).sum()
    sla_mask[0, 1:-1, 1:-1]
    sla_mask[0]

    att_mask[0, 1:7]

    ids_in_range(head_maps[0], 1, 10)
    head_maps[0]
    head
    max_distance = 1
