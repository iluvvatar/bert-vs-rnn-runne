import os
from pathlib import Path
from transformers import AutoModel

from mlpipeline.utils import PathLike


def download_hf_model(model_name: str,
                      save_to_folder: PathLike):
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_to_folder)


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    download_hf_model('DeepPavlov/rubert-base-cased',
                      home_dir / 'models/DeepPavlov-rubert-base-cased')
