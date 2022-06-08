import os
from pathlib import Path

from mlpipeline.datasets.nlp import Gazeta


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    datasets_dir = home_dir / 'datasets'
    path = datasets_dir / 'Gazeta' / 'HuggingFaceLocal'
    dataset = Gazeta.load_from_hub()
    dataset.save(path)
