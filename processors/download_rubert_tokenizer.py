import os
from transformers import BertTokenizer
from pathlib import Path

from mlpipeline.utils import PathLike


def download_tokenizer(name: str,
                       folder_path: PathLike):
    tokenizer = BertTokenizer.from_pretrained(name)
    path = Path(folder_path)
    if not path.exists():
        path.mkdir(parents=True)
    tokenizer.save_pretrained(folder_path)


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    download_tokenizer('DeepPavlov/rubert-base-cased',
                       home_dir / 'models' / 'tokenizers')
