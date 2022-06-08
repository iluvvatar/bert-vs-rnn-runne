import torch
from pathlib import Path
import os
from time import time
import json
from torch.utils.data import DataLoader

from mlpipeline.datasets.nlp import RuNNE
from mlpipeline.processors.nlp.collators import PaddingCollator
from mlpipeline.models.nlp.ner import BertForNER
from mlpipeline.models.nlp.ner import ConvLSTM
from mlpipeline.models.nlp.ner import ConvSRU
from mlpipeline.models.nlp.ner import ConvSRUpp

from tqdm import tqdm


def main():
    bert_path = Path(os.getenv('HOME')) / 'models' / 'DeepPavlov-rubert-base-cased'
    models = [
        BertForNER(bert_path, n_ent_types=29, n_classes=5),
        ConvLSTM(n_ent_types=29,
                 n_classes=5,
                 cnn_layers=2,
                 cnn_kernels=[1, 3, 5],
                 rnn_layers=4,
                 hid_size=768//2,
                 dropout=0.1),
        ConvSRU(n_ent_types=29,
                n_classes=5,
                cnn_layers=2,
                cnn_kernels=[1, 3, 5],
                rnn_layers=4,
                hid_size=768 // 2,
                dropout=0.1),
        ConvSRUpp(n_ent_types=29,
                  n_classes=5,
                  cnn_layers=2,
                  cnn_kernels=[1, 3, 5],
                  rnn_layers=4,
                  hid_size=768 // 2,
                  dropout=0.1)
    ]
    devices = ['cpu', 'cuda']
    results = [{'name': model.name} for model in models]

    runne_path = Path(os.getenv('HOME')) / 'datasets' / 'RuNNE' / 'Preprocessed'
    runne_bert_path = Path(os.getenv('HOME')) / 'datasets' / 'RuNNE' / 'PreprocessedForBert'
    output_file_path = Path(os.getenv('SCRATCH_DIR')) / 'models' / 'inference_time.jsonl'
    if not output_file_path.parent.exists():
        output_file_path.mkdir(parents=True)
    runne_test = RuNNE.load(runne_path)['test']
    runne_bert_test = RuNNE.load(runne_bert_path)['test']
    collator = PaddingCollator(
        collate_columns=['tokens_ids', 'attention_mask', 'labels_ids'],
        pad_value=0,
        padding_type='longest'
    )

    for i, model_ in enumerate(models):
        model_.eval()
        model_.freeze_embeddings()
        results[i]['parameters_number'] = sum(p.numel()
                                              for p in model_.parameters()
                                              if p.requires_grad)
        if 'bert' in model_.name.lower():
            loader = DataLoader(runne_bert_test,
                                batch_size=1,
                                collate_fn=collator.collate)
        else:
            loader = DataLoader(runne_test,
                                batch_size=1,
                                collate_fn=collator.collate)
        with torch.no_grad():
            for device in devices:
                model = model_.to(device)
                total_time = 0
                for batch in tqdm(loader, desc=f'{model.name}-{device}'):
                    tokens_ids = batch['tokens_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_time = time()
                    logits = model(tokens_ids=tokens_ids,
                                   attention_mask=attention_mask)
                    total_time += time() - start_time
                results[i][f'{device}-time-per-sample'] = total_time / len(runne_test)
        print(results)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for result in results:
                print(json.dumps(result), file=f)


if __name__ == '__main__':
    main()
