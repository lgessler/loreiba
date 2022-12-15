################################################################################
# Subtree sampling methods
################################################################################
import random
from typing import Any, Dict, List

from tango.common import FromParams, Registrable


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
        min_subtree_size: int = 2,
        max_replacements: int = 3,
        max_retry: int = 10,
        subtree_sampling_method: SubtreeSamplingMethod = AllSubtreeSamplingMethod(),
    ):
        self.max_negative_per_subtree = max_negative_per_subtree
        self.min_subtree_size = min_subtree_size
        self.max_replacements = max_replacements
        self.max_retry = max_retry
        self.x = x
        self.subtree_sampling_method = subtree_sampling_method
