from pathlib import Path

from mlpipeline.trainers.utils.history import History


ent_types = {
    'AGE': 692,
    'AWARD': 457,
    'CITY': 1342,
    'COUNTRY': 2977,
    'CRIME': 217,
    'DATE': 2807,
    'DISEASE': 89,
    'DISTRICT': 123,
    'EVENT': 3600,
    'FACILITY': 434,
    'FAMILY': 31,
    'IDEOLOGY': 343,
    'LANGUAGE': 51,
    'LAW': 458,
    'LOCATION': 336,
    'MONEY': 214,
    'NATIONALITY': 460,
    'NUMBER': 1256,
    'ORDINAL': 672,
    'ORGANIZATION': 4744,
    'PENALTY': 51,
    'PERCENT': 89,
    'PERSON': 5480,
    'PRODUCT': 291,
    'PROFESSION': 5480,
    'RELIGION': 118,
    'STATE_OR_PROVINCE': 455,
    'TIME': 201,
    'WORK_OF_ART': 127
}
ent_types = sorted(ent_types.keys(), key=lambda ent_type: ent_types[ent_type], reverse=True)
metrics = ['f1-macro', 'f1-main', 'f1-few-shot'] + [f'f1-{e}' for e in ent_types]


def main():
    root = Path('D:/Research/RNNvsBERT/models/result')
    ft_path = root / 'fine_tuning_runne'
    gz_path = root / 'gazeta'
    rn_path = root / 'runne'
    rn_key = 'usual NER (RuNNE)'
    gz_key = 'distillation (Gazeta)'
    ft_key = 'fine-tuning (RuNNE)'
    keys = [gz_key, ft_key, rn_key]
    history_paths = {
        'CLSTM': {ft_key: ft_path / 'ConvLSTM_epoch=4_f1-macro=0.65607' / 'history.json',
                  gz_key: gz_path / 'ConvLSTM_epoch=40_f1-macro=0.64941' / 'history.json',
                  rn_key: rn_path / 'ConvLSTM_epoch=30_f1-macro=0.52193' / 'history.json'},
        'CSRU': {ft_key: ft_path / 'ConvSRU_epoch=4_f1-macro=0.63647' / 'history.json',
                 gz_key: gz_path / 'ConvSRU_epoch=40_f1-macro=0.61654' / 'history.json',
                 rn_key: rn_path / 'ConvSRU_epoch=20_f1-macro=0.55312' / 'history.json'},
        'CSRU++': {ft_key: ft_path / 'ConvSRUpp_epoch=4_f1-macro=0.66051' / 'history.json',
                   gz_key: gz_path / 'ConvSRUpp_epoch=40_f1-macro=0.64315' / 'history.json',
                   rn_key: rn_path / 'ConvSRUpp_epoch=20_f1-macro=0.55573' / 'history.json'},
    }

    with open(root / 'f1.txt', 'w') as f:
        for model, paths in history_paths.items():
            print(model, file=f)
            print('==========================================================', file=f)
            for key in keys:
                history = History.load(paths[key])
                print(key, file=f)
                print('----------------', file=f)
                for metric in metrics:
                    value = history.metrics[metric][-1]
                    print(f'{metric:<30}{round(value, 3)}', file=f)
                print(file=f)
            print('==========================================================', file=f)


if __name__ == '__main__':
    main()
