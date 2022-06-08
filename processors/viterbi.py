import os
from pathlib import Path

from mlpipeline.datasets import NEREL
from mlpipeline.processors.labelizers import BILOULabelizer
from mlpipeline.processors.collators import MaxLenSplitCollator
from mlpipeline.processors.prediction_postprocessors import Viterbi


def main():
    home_dir = Path(os.getenv('HOME'))
    dataset_path = home_dir / 'datasets/NEREL/BondModelPredictedLogits'
    dataset = NEREL.load_from_disk(dataset_path)

    viterbi = Viterbi(logits_column='logits',
                      word_tokens_indices_column='word_tokens_indices',
                      first_subword_transition_probs=BILOULabelizer.first_subword_transition_probs,
                      middle_subword_transition_probs=BILOULabelizer.middle_subword_transition_probs,
                      last_subword_transition_probs=BILOULabelizer.last_subword_transition_probs,
                      word_transition_probs=BILOULabelizer.word_transition_probs)
    dataset = viterbi.postprocess(dataset, use_cached=True)
    print(dataset)

    dataset.save(home_dir / 'datasets/NEREL/BondModelDecodedPredictions')


if __name__ == '__main__':
    main()
