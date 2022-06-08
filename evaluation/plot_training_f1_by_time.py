import os
from matplotlib import pyplot as plt
from pathlib import Path

from mlpipeline.trainers.utils.history import History


def get_time_array(history: History, start_time: float = 0):
    time = [start_time]
    for start, end in zip(history.start_timestamp[1:],
                          history.end_timestamp[1:]):
        duration = (end - start).total_seconds() / 3600
        time.append(time[-1] + duration)
    return time


def main():
    figsize = (9, 3)
    root = Path(os.getenv('HOME')) / 'models' / 'result'
    ft_path = root / 'fine_tuning_runne'
    gz_path = root / 'gazeta'
    rn_path = root / 'runne'
    rn_key = 'usual NER'
    gz_key = 'distillation'
    ft_key = 'fine-tuning'
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
    colors = {ft_key: 'green',
              gz_key: 'orange',
              rn_key: 'royalblue'}

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (model, paths) in zip(axes, history_paths.items()):
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Training hours')
        ax.set_ylabel('F1-macro')
        ax.set_title(model)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # RuNNE
        history = History.load(paths[rn_key])
        f1 = history.metrics['f1-macro']
        time = get_time_array(history, 0)
        ax.plot(time, f1, label=rn_key, color=colors[rn_key])

        # Gazeta
        history = History.load(paths[gz_key])
        f1 = history.metrics['f1-macro']
        time = get_time_array(history, 0)
        if model == 'CSRU':
            start_idx = 5
            f1 = f1[start_idx:]
            time = get_time_array(history, )[start_idx:]
            start = time[0]
            for i in range(len(time)):
                time[i] -= start
        ax.plot(time, f1, label=gz_key, color=colors[gz_key])

        # Fine-tuning
        history = History.load(paths[ft_key])
        f1 = history.metrics['f1-macro']
        time = get_time_array(history, time[-1])
        ax.plot(time, f1, label=ft_key, color=colors[ft_key])
        ax.legend(loc='lower right')

    fig.show()
    fig.savefig(root / 'training.png')


if __name__ == '__main__':
    main()
