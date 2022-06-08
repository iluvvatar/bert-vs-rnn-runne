from pathlib import Path
import numpy as np
from datasets.arrow_dataset import Batch
import os
import json

from mlpipeline.datasets.nlp import RuNNE
from mlpipeline.processors.nlp.collators import PaddingCollator
from mlpipeline.processors.nlp.labelizers import BILOULabelizer
from mlpipeline.processors.nlp.prediction_postprocessors.viterbi import Viterbi
from mlpipeline.datasets.nlp.units import Entity
from mlpipeline.evaluators import EvaluatorNER

from bond_model.ner import load_ner


def main():
    home_dir = Path(os.getenv('HOME'))
    datasets_dir = home_dir / 'datasets'
    models_dir = home_dir / 'models'

    runne_path = datasets_dir / 'RuNNE' / 'PreprocessedForBert'
    bond_model_path = models_dir / 'dp_rubert_from_siamese'
    tokenizer_path = models_dir / 'tokenizers' / 'DeepPavlov-rubert-base-cased'
    config_path = models_dir / 'DeepPavlov-rubert-base-cased'

    output_dir = Path(os.getenv('SCRATCH_DIR')) / 'datasets' / 'runne' / 'bond_predictions'

    runne = RuNNE.load(runne_path)

    bond_model, bond_tokenizer, max_sent_len, ent_types = load_ner(
        bond_model_path, tokenizer_path, config_path)
    print(bond_model)

    padder = PaddingCollator(pad_value=0,
                             padding_type='max_length',
                             max_len=128,
                             collate_columns=['tokens_ids'])

    def predict_logits_map_fn(batch: Batch):
        tokens_ids = padder.collate(batch)['tokens_ids'].numpy()
        output = bond_model(tokens_ids)
        output = np.stack(output, axis=0).transpose((1, 2, 0, 3))
        logits = []
        for i, out in enumerate(output):
            length = len(batch['tokens_ids'][i])
            logits.append(out[:length])
        return {'logits': logits}

    runne = runne.map(predict_logits_map_fn, batched=True)

    viterbi = Viterbi(logits_column='logits',
                      word_tokens_indices_column='word_tokens_indices',
                      out_predicted_labels_ids_column='predicted_labels_ids',
                      first_subword_transition_probs=BILOULabelizer.first_subword_transition_probs,
                      middle_subword_transition_probs=BILOULabelizer.middle_subword_transition_probs,
                      last_subword_transition_probs=BILOULabelizer.last_subword_transition_probs,
                      word_transition_probs=BILOULabelizer.word_transition_probs,
                      initial_state=0,
                      pad_label_id=0)
    labelizer = BILOULabelizer(text_column='text',
                               tokens_column='tokens',
                               tokens_spans_column='spans',
                               entities_column='entities',
                               predicted_labels_ids_column='predicted_labels_ids',
                               entity_types=runne.entity_types,
                               out_predicted_entities_column='predicted_entities',
                               out_labels_ids_column='labels_ids',
                               entities_deserialize_fn=Entity.from_str,
                               entities_serialize_fn=Entity.to_str)
    runne = viterbi.postprocess(runne, use_cached=False)
    runne = labelizer.postprocess(runne, use_cached=False)

    runne.save(output_dir)

    evaluator = EvaluatorNER(real_entities_column='entities',
                             pred_entities_column='predicted_entities',
                             entity_types=runne.entity_types,
                             entities_deserialize_fn=Entity.from_str)

    result = {}
    for split in runne:
        result[split] = evaluator.evaluate(runne[split])
    with open(output_dir / 'eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main()
