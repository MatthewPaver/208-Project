from tensorflow.data import Dataset
import tensorflow_datasets as tfds
def load_dataset() -> Dataset:
    ds = tfds.load('RockPaperScissors', split='train', as_supervised=True, shuffle_files=True)
    return ds