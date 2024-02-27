from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def load_dataset(path: str) -> Dataset:
    #TODO: Load in all data instances and labels, also complete any preprocessing neccessary
    return

def create_dataloader(data: Dataset, batch_size: int) -> DataLoader:
    #TODO: Wrap the data in a dataloader in batches of size that was given
    return