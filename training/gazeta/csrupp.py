import torch

from mlpipeline.models.nlp.ner import ConvSRUpp
from training.gazeta.pipeline import gazeta_pipeline


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvSRUpp(n_ent_types=29,
                      n_classes=5,
                      cnn_layers=2,
                      cnn_kernels=[1, 3, 5],
                      rnn_layers=4,
                      hid_size=768//2,
                      head_layers=1,
                      dropout=0.1).to(device)
    checkpoint = 'ConvSRUpp_epoch=10_f1-macro=0.57072'
    gazeta_pipeline(model, device, experiment_name='v7.0.2', checkpoint_name=checkpoint)


if __name__ == '__main__':
    main()
