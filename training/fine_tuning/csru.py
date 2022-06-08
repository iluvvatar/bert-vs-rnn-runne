import os
from pathlib import Path
import torch

from mlpipeline.models.nlp.ner import ConvSRU
from training.fine_tuning.pipeline import fine_tuning_pipeline


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvSRU(n_ent_types=29,
                    n_classes=5,
                    cnn_layers=2,
                    cnn_kernels=[1, 3, 5],
                    rnn_layers=4,
                    hid_size=768//2,
                    head_layers=1,
                    dropout=0.1).to(device)
    pretrained_gazeta_model_path = Path(os.getenv('SCRATCH_DIR')) / 'models' \
                                   / model.name / 'gazeta' / 'v7.0.0' \
                                   / 'ConvSRU_epoch=40_f1-macro=0.61654' / 'model.pt'
    checkpoint = 'ConvSRU_epoch=6_f1-macro=0.63951'
    fine_tuning_pipeline(model, pretrained_gazeta_model_path, device,
                         experiment_name='v7.0.0', checkpoint_name=checkpoint)


if __name__ == '__main__':
    main()
