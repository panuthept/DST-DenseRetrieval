import random
random.seed(313)
from dataclasses import dataclass
from typing import Union, List

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from tevatron_dst.arguments import ModelArguments
from textattack.transformations import WordSwapNeighboringCharacterSwap, \
    WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion, \
    WordSwapRandomCharacterSubstitution, WordSwapQWERTY
from textattack.constraints.pre_transformation import MinWordLength, StopwordModification
from .modeling import CharacterIndexer
from .query_augmentation import QueryAugmentation

from .arguments import DataArguments
from .trainer import DenseTrainer

import logging
logger = logging.getLogger(__name__)


STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                      "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                      'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                      'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                      'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                      'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                      'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                      "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                      "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                      'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                      "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                      'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])


class FixWordSwapQWERTY(WordSwapQWERTY):
    def _get_replacement_words(self, word):
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = len(word) - (1 + self.skip_last_char)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            i = random.randrange(start_idx, end_idx + 1)
            if len(self._get_adjacent(word[i])) == 0:
                candidate_word = (
                    word[:i] + random.choice(list(self._keyboard_adjacency.keys())) + word[i + 1:]
                )
            else:
                candidate_word = (
                    word[:i] + random.choice(self._get_adjacent(word[i])) + word[i + 1:]
                )
            candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx + 1):
                for swap_key in self._get_adjacent(word[i]):
                    candidate_word = word[:i] + swap_key + word[i + 1 :]
                    candidate_words.append(candidate_word)

        return candidate_words


class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            path_to_data: Union[List[str], datasets.Dataset],
            tokenizer: PreTrainedTokenizer,
            cache_dir: str,
            trainer: DenseTrainer = None,
            character_query_encoder: bool = False,
    ):
        if isinstance(path_to_data, datasets.Dataset):
            self.train_data = path_to_data
        else:
            self.train_data = datasets.load_dataset(
                'json',
                data_files=path_to_data,
                ignore_verifications=False,
                cache_dir=cache_dir
            )['train']

        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

        constraints = [MinWordLength(3), StopwordModification(STOPWORDS)]
        transformations=[
            WordSwapRandomCharacterInsertion(),
            WordSwapRandomCharacterSubstitution(),
            WordSwapRandomCharacterDeletion(),
            WordSwapNeighboringCharacterSwap(),
            FixWordSwapQWERTY()
        ]
        self.augmenter = QueryAugmentation(
            expansion_size=self.data_args.query_augmentation_size, 
            constraints=constraints,
            transformations=transformations
        )

        self.character_query_encoder = character_query_encoder
        if self.character_query_encoder:
            self.character_indexer = CharacterIndexer()

    def _character_bert_tokenize(self, text_encoding, decode=False):
        if decode:
            text_encoding = self.tok.decode(text_encoding)

        x = self.tok.basic_tokenizer.tokenize(text_encoding)
        x = ['[CLS]', *x, '[SEP]']
        return x

    def create_one_example(self, text_encoding: Union[List[int], str], is_query=False):
        if self.character_query_encoder:
            item = self._character_bert_tokenize(text_encoding, decode=True)
        else:
            item = self.tok.encode_plus(
                text_encoding,
                truncation='only_first',
                max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        return item

    def create_typo_examples(self, text_encoding: Union[List[int], str], is_query=True):
        text_encoding = self.tok.decode(text_encoding)
        text_encodings = self.augmenter.augment(text_encoding)
        text_encodings = [text_encoding.lower() for text_encoding in text_encodings]

        items = []
        for text_encoding in text_encodings:
            if self.character_query_encoder:
                item = self._character_bert_tokenize(text_encoding, decode=False)
            else:
                item = self.tok.encode_plus(
                    text_encoding,
                    truncation='only_first',
                    max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
            items.append(item)
        return items

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']
        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        encoded_passages.append(self.create_one_example(pos_psg))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(self.create_one_example(neg_psg))

        encoded_typo_queries = self.create_typo_examples(qry, is_query=True)
        return encoded_query, encoded_typo_queries, encoded_passages


class EncodeDataset(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(self, path_to_json: Union[List[str], datasets.Dataset], tokenizer: PreTrainedTokenizer,
                 model_args: ModelArguments,
                 max_len=128):
        if isinstance(path_to_json, datasets.Dataset):
            self.encode_data = path_to_json
        else:
            self.encode_data = []
            for path in path_to_json:
                with open(path, 'r') as f:
                    for line in f:
                        text_id, text = line.strip().split('\t')
                        self.encode_data.append({'text_id': text_id, 'text': text})
        self.tok = tokenizer
        self.max_len = max_len
        self.model_args = model_args
        self.decode = True

    def _character_bert_tokenize(self, text, decode=False):
        if decode:  # lazy fix for now
            text = self.tok.decode(text)

        x = self.tok.basic_tokenizer.tokenize(text)
        x = ['[CLS]', *x, '[SEP]']
        return x

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item):
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)

        if self.model_args.character_query_encoder:
            encoded_text = self._character_bert_tokenize(text, self.decode)
        else:
            encoded_text = self.tok.encode_plus(
                text,
                max_length=self.max_len,
                truncation='only_first',
                padding=False,
                return_token_type_ids=False,
            )
        return text_id, encoded_text


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return q_collated, d_collated

@dataclass
class CharacterQPCollator(QPCollator):
    indexer: CharacterIndexer = CharacterIndexer()

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.indexer.as_padded_tensor(
            qq,
            maxlen=self.max_q_len
        )
        d_collated = self.indexer.as_padded_tensor(
            dd,
            maxlen=self.max_p_len
        )

        return q_collated, d_collated

@dataclass
class TypoQPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]   # (batch_size, words_num)
        tqq = [f[1] for f in features]  # (batch_size, words_num)
        dd = [f[2] for f in features]   # (batch_size, words_num)

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(tqq[0], list):
            tqq = sum(tqq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        tq_collated = self.tokenizer.pad(
            tqq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return q_collated, tq_collated, d_collated


@dataclass
class TypoCharacterQPCollator(TypoQPCollator):
    indexer: CharacterIndexer = CharacterIndexer()

    def __call__(self, features):
        qq = [f[0] for f in features]   # (batch_size, words_num, chars_num)
        tqq = [f[1] for f in features]  # (batch_size, queries_num, words_num, chars_num)
        dd = [f[2] for f in features]   # (batch_size, passages_num, words_num, chars_num)

        if isinstance(tqq[0], list):
            tqq = sum(tqq, [])          # (batch_size * queries_num, words_num, chars_num)
        if isinstance(dd[0], list):
            dd = sum(dd, [])            # (batch_size * passages_num, words_num, chars_num)

        q_collated = self.indexer.as_padded_tensor(
            qq,
            maxlen=self.max_q_len
        )
        tq_collated = self.indexer.as_padded_tensor(
            tqq,
            maxlen=self.max_q_len
        )
        d_collated = self.indexer.as_padded_tensor(
            dd,
            maxlen=self.max_p_len
        )
        return q_collated, tq_collated, d_collated


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features


@dataclass
class EncodeCharacterCollator(DataCollatorWithPadding):
    indexer: CharacterIndexer = CharacterIndexer()

    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]

        collated_features = self.indexer.as_padded_tensor(
            text_features,
            maxlen=self.max_length
        )
        return text_ids, collated_features