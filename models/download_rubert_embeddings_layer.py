import os
import torch
from transformers import BertModel
from pathlib import Path


def main():
    home_dir = Path(os.getenv('HOME'))
    path = home_dir / 'models/DeepPavlov-rubert-base-cased_input_embeddings_layer'
    bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
    embeddings = bert.get_input_embeddings()
    if not path.exists():
        path.mkdir(parents=True)
    torch.save(embeddings.state_dict(), path / 'state_dict.pt')


if __name__ == '__main__':
    main()
