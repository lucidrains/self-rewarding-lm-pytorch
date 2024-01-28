from functools import wraps
from typing import Type, Any
from torch.utils.data import Dataset

def always(val):
    def decorator(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            if callable(val):
                return val()

            return val
        return inner
    return decorator

def create_mock_dataset(
    length: int,
    output: Any
) -> Dataset:

    class MockDataset(Dataset):
        def __len__(self):
            return length

        def __getitem__(self, idx):
            if callable(output):
                return output()

            return output

    return MockDataset()
