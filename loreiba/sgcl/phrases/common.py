from tango.common import FromParams


class PhraseSgclConfig(FromParams):
    def __init__(
        self,
        min_subtree_token_count: int = 2,
        max_subtree_height: int = 2,
        negative_per_positive: int = 5,
        max_subtrees_per_sentence: int = 10,
        temperature: float = 0.1,
    ):
        self.min_subtree_token_count = min_subtree_token_count
        self.max_subtree_height = max_subtree_height
        self.negative_per_positive = negative_per_positive
        self.max_subtrees_per_sentence = max_subtrees_per_sentence
        self.temperature = temperature
