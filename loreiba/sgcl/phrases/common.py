from tango.common import FromParams


class PhraseSgclConfig(FromParams):
    def __init__(
        self,
        min_subtree_token_count: int = 2,
        max_subtree_height: int = 2,
        negative_per_positive: int = 5,
    ):
        self.min_subtree_token_count = min_subtree_token_count
        self.max_subtree_height = max_subtree_height
        self.negative_per_positive = negative_per_positive
