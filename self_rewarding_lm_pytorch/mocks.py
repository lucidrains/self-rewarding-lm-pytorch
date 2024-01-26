from typing import Type, Any
from torch.utils.data import Dataset

def create_mock_dataset(
    length: int,
    output: Any
) -> Type[Dataset]:

    class MockDataset(Dataset):
        def __len__(self):
            return length

        def __getitem__(self, idx):
            if callable(output):
                return output()

            return output

    return MockDataset
