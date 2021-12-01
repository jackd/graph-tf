import typing as tp


class DataSplit(tp.NamedTuple):
    train_data: tp.Iterable
    validation_data: tp.Optional[tp.Iterable]
    test_data: tp.Optional[tp.Iterable]


def data_split(
    train_data: tp.Iterable,
    validation_data: tp.Optional[tp.Iterable] = None,
    test_data: tp.Optional[tp.Iterable] = None,
) -> DataSplit:
    return DataSplit(train_data, validation_data, test_data)
