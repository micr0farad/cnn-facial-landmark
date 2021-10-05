import typing

import numpy as np
import torch
from tfrecord import tfrecord_iterator, example_pb2, extract_feature_dict
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import IterableDataset


class IterableTFRecordDataset(TFRecordDataset):
    cache = []

    def __init__(
        self,
        data_path: str,
        index_path: typing.Union[str, None],
        description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
        shuffle_queue_size: typing.Optional[int] = None,
        transform: typing.Callable[[dict], typing.Any] = None,
        sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
        compression_type: typing.Optional[str] = None,
    ) -> None:
        super().__init__(
            data_path,
            index_path,
            description,
            shuffle_queue_size,
            transform,
            sequence_description,
            compression_type,
        )
        self._fill_cache()

    def _fill_cache(self):
        self.cache = [feature_dict for feature_dict in self]

    def __getitem__(self, item):
        return self.cache[item]

    def __len__(self):
        return len(self.cache)
