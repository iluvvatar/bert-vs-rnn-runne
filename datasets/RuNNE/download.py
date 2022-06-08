import os
from pathlib import Path

from mlpipeline.utils import PathLike
from mlpipeline.datasets.nlp import RuNNE


def download(folder_path: PathLike):
    folder_path = Path(folder_path)
    dataset = RuNNE.load_from_hub()
    dataset.save(folder_path)


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets' / 'RuNNE'
    download(dataset_dir / 'HuggingFaceLocal')
