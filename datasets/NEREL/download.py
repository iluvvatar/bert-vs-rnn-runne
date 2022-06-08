import os
from pathlib import Path

from mlpipeline.datasets.nlp import NEREL


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    datasets_dir = home_dir / 'datasets'
    path = datasets_dir / 'NEREL' / 'HuggingFaceLocal'
    dataset = NEREL.load_from_hub()
    dataset.save(path)
