import torch

from mlpipeline.models.nlp.ner import ConvSRU
from training.runne.pipeline import runne_pipeline


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
    runne_pipeline(model, device, experiment_name='v8.0.0')


if __name__ == '__main__':
    main()
