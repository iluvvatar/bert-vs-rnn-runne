import os
from pathlib import Path

from mlpipeline.datasets.nlp import RuNNE
from mlpipeline.datasets.nlp.units import Entity
from mlpipeline.processors.nlp.sentenizers import NatashaSentenizer
from mlpipeline.processors.nlp.tokenizers import RuBERTTokenizer
from mlpipeline.processors.nlp.labelizers import BILOULabelizer


def main():
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets' / 'RuNNE'
    dataset_path = dataset_dir / 'HuggingFaceLocal'
    save_path = dataset_dir / 'Preprocessed'
    tokenizer_path = 'DeepPavlov/rubert-base-cased'

    dataset = RuNNE.load(dataset_path)

    sentenizer = NatashaSentenizer(text_column='text',
                                   doc_id_column='id',
                                   entities_column='entities',
                                   remove_columns=['id'],
                                   out_text_column='text',
                                   out_start_column='start',
                                   out_stop_column='stop',
                                   out_doc_id_column='doc_id',
                                   out_entities_column='entities',
                                   entities_deserialize_fn=Entity.from_str,
                                   entities_serialize_fn=Entity.to_str)
    tokenizer = RuBERTTokenizer(text_column='text',
                                pretrained_tokenizer_path=tokenizer_path,
                                out_tokens_column='tokens',
                                out_spans_column='spans',
                                out_tokens_ids_column='tokens_ids',
                                out_word_tokens_indices_column='word_tokens_indices',
                                out_attention_mask_column='attention_mask',
                                out_token_type_ids_column='token_type_ids')
    labelizer = BILOULabelizer(text_column='text',
                               tokens_column='tokens',
                               tokens_spans_column='spans',
                               entities_column='entities',
                               predicted_labels_ids_column='predicted_labels_ids',
                               out_labels_column='labels',
                               out_labels_ids_column='labels_ids',
                               out_predicted_entities_column='predicted_entities',
                               entity_types=dataset.entity_types,
                               special_tokens=tokenizer.special_tokens,
                               entities_deserialize_fn=Entity.from_str,
                               entities_serialize_fn=Entity.to_str)

    dataset = sentenizer.preprocess(dataset, use_cached=False)
    dataset = tokenizer.preprocess(dataset, use_cached=False)
    dataset = labelizer.preprocess(dataset, use_cached=False)

    dataset.save(save_path)


if __name__ == '__main__':
    main()
