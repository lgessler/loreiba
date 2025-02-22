import os
from logging import getLogger
from typing import Iterator, List, Literal

import more_itertools as mit
from datasets import IterableDataset
from tokenizers import Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
# https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from transformers import (BertTokenizer, BertTokenizerFast, PreTrainedTokenizerFast, RobertaTokenizer,
                          RobertaTokenizerFast)

logger = getLogger(__name__)


def write_vocab(tokenizer: Tokenizer, serialization_dir: str):
    vocab = [(w, i) for w, i in tokenizer.get_vocab().items()]
    vocab = sorted(vocab, key=lambda x: x[1])
    assert [i for _, i in vocab] == list(range(vocab[-1][1] + 1)), "Vocabulary is not monotonic!"
    words = "\n".join([w for w, _ in vocab]) + "\n"
    with open(os.path.join(serialization_dir, "vocab.txt"), "w") as f:
        f.write(words)
    logger.info("Wrote vocab to" + serialization_dir)


def count_word_types(sentences: List[str]):
    vocab = set()
    for sentence in sentences:
        for token in sentence.split(" "):
            vocab.add(token)
    return len(vocab)


def train_tokenizer_generator(
    dataset: IterableDataset, vocab_size, serialize_path: str = "", tokenization_type="wordpiece"
) -> Tokenizer:
    # Some reference points for train splits:
    # - Greek: 9_058_227 words, 478_376 types
    # - Wolof: 517_237 words, 59_137 types
    # - Coptic: 970_642 words, 14_457 types
    # - Uyghur: 2_401_445 words, 368_585 types
    # - Maltese: 2_113_223 words, 319_083 types

    cls_token = "[CLS]"
    sep_token = "[SEP]"
    unk_token = "[UNK]"
    pad_token = "[PAD]"
    mask_token = "[MASK]"
    special_tokens = [pad_token, cls_token, sep_token, unk_token, mask_token]

    models = {"wordpiece": WordPiece, "bpe": BPE}
    if tokenization_type not in models:
        raise Exception(f"Unknown model type {tokenization_type}. Valid models: {list(models.keys())}")

    tokenizer = Tokenizer(models[tokenization_type](unk_token="[UNK]"))
    tokenizer.normalizer = Sequence(
        [
            NFD(),
            Lowercase(),
            # StripAccents()
        ]
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single=f"{cls_token} $A {sep_token}",
        pair=f"{cls_token} $A {sep_token} $B:1 {sep_token}:1",
        special_tokens=[
            (cls_token, 1),
            (sep_token, 2),
        ],
    )

    trainers = {"wordpiece": WordPieceTrainer, "bpe": BpeTrainer}
    if tokenization_type not in trainers:
        raise Exception(f"Unknown model type {tokenization_type}. Valid models: {list(trainers.keys())}")

    logger.info("Beginning tokenizer training")
    trainer = trainers[tokenization_type](vocab_size=vocab_size, special_tokens=special_tokens)

    def sentences():
        for s in dataset:
            yield " ".join(s["tokens"])

    tokenizer.train_from_iterator(sentences(), trainer=trainer)
    if serialize_path:
        full_tokenizer = BertTokenizerFast(
            tokenizer_object=tokenizer,
            cls_token=cls_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            bos_token=cls_token,
            eos_token=sep_token,
        )
        full_tokenizer.save_pretrained(serialize_path)
        write_vocab(tokenizer, serialize_path)
    return tokenizer


def train_tokenizer(
    sentences: List[str], serialize_path: str = "", tokenization_type="wordpiece", vocab_size=None
) -> Tokenizer:
    # Some reference points for train splits:
    # - Greek: 9_058_227 words, 478_376 types
    # - Wolof: 517_237 words, 59_137 types
    # - Coptic: 970_642 words, 14_457 types
    # - Uyghur: 2_401_445 words, 368_585 types
    # - Maltese: 2_113_223 words, 319_083 types
    word_type_count = count_word_types(sentences)
    if vocab_size is None:
        vocab_size = 8_000 + max(0, int((word_type_count - 15000) * (6_000 / 450_000)))

    cls_token = "[CLS]"
    sep_token = "[SEP]"
    unk_token = "[UNK]"
    pad_token = "[PAD]"
    mask_token = "[MASK]"
    special_tokens = [pad_token, cls_token, sep_token, unk_token, mask_token]

    models = {"wordpiece": WordPiece, "bpe": BPE}
    if tokenization_type not in models:
        raise Exception(f"Unknown model type {tokenization_type}. Valid models: {list(models.keys())}")

    tokenizer = Tokenizer(models[tokenization_type](unk_token="[UNK]"))
    tokenizer.normalizer = Sequence(
        [
            NFD(),
            Lowercase(),
            # StripAccents()
        ]
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single=f"{cls_token} $A {sep_token}",
        pair=f"{cls_token} $A {sep_token} $B:1 {sep_token}:1",
        special_tokens=[
            (cls_token, 1),
            (sep_token, 2),
        ],
    )

    trainers = {"wordpiece": WordPieceTrainer, "bpe": BpeTrainer}
    if tokenization_type not in trainers:
        raise Exception(f"Unknown model type {tokenization_type}. Valid models: {list(trainers.keys())}")

    trainer = trainers[tokenization_type](vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    if serialize_path:
        full_tokenizer = BertTokenizerFast(
            tokenizer_object=tokenizer,
            cls_token=cls_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            bos_token=cls_token,
            eos_token=sep_token,
        )
        full_tokenizer.save_pretrained(serialize_path)
        write_vocab(tokenizer, serialize_path)
    return tokenizer


def train_bert_tokenizer(
    sentences: List[str],
    serialize_path: str,
    vocab_size: int = 10_000,
    tokenization_type: Literal["bpe", "wordpiece"] = "wordpiece",
) -> BertWordPieceTokenizer:
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )
    tokenizer.train_from_iterator(
        sentences,
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=500,
        wordpieces_prefix="##",
    )

    # Save the files--first write out the vocab, then use BertTokenizer's save_pretrained
    tokenizer.save_model(serialize_path)
    bert_tokenizer = BertTokenizer.from_pretrained(serialize_path)
    bert_tokenizer.save_pretrained(serialize_path)
    # os.rename(
    #    serialize_path + os.sep + "tokenizer_config.json",
    #    serialize_path + os.sep + "config.json"
    # )
    return bert_tokenizer


def simple_train_tokenizer(sentences: List[List[str]], output_path: str, vocab_size: int = 10_000):
    sentences = [" ".join(s) for s in sentences]
    batched_sentences = mit.chunked(sentences, 32)
    t = BertTokenizerFast.from_pretrained("bert-base-cased")
    t = t.train_new_from_iterator(batched_sentences, vocab_size=vocab_size)
    t.save_pretrained(output_path)
    return t
