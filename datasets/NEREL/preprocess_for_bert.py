import os
from pathlib import Path

from mlpipeline.datasets.nlp import NEREL
from mlpipeline.processors.nlp.sentenizers import NatashaSentenizer
from mlpipeline.processors.nlp.tokenizers import RuBERTTokenizer
from mlpipeline.processors.nlp.labelizers import BILOULabelizer
from mlpipeline.processors.nlp.collators import MaxLenSplitCollator


def main():
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets' / 'NEREL'
    dataset_path = dataset_dir / 'HuggingFaceLocal'
    save_path = dataset_dir / 'PreprocessedForBert'
    tokenizer_path = 'DeepPavlov/rubert-base-cased'

    dataset = NEREL.load(dataset_path)

    sentenizer = NatashaSentenizer(text_column='text',
                                   doc_id_column='id',
                                   entities_column='entities',
                                   remove_columns=['relations', 'links', 'id'],
                                   out_text_column='text',
                                   out_start_column='start',
                                   out_stop_column='stop',
                                   out_doc_id_column='doc_id')
    tokenizer = RuBERTTokenizer(text_column='text',
                                pretrained_tokenizer_path=tokenizer_path,
                                out_tokens_column='tokens',
                                out_spans_column='spans',
                                out_tokens_ids_column='tokens_ids',
                                out_word_tokens_indices_column='out_word_tokens_indices_column',
                                out_attention_mask_column='attention_mask',
                                out_token_type_ids_column='token_type_ids')
    labelizer = BILOULabelizer(text_column='text',
                               tokens_column='tokens',
                               tokens_spans_column='spans',
                               entities_column='entities',
                               predicted_labels_ids_column='predicted_labels_ids',
                               out_labels_ids_column='labels_ids',
                               out_predicted_entities_column='predicted_entities',
                               entity_types=dataset.entity_types,
                               special_tokens=tokenizer.special_tokens)
    collator = MaxLenSplitCollator(
        collate_columns=['tokens', 'spans', 'tokens_ids', 'attention_mask',
                         'token_type_ids', 'labels_ids'],
        pk_columns=['doc_id', 'start'],
        unite_columns=['logits'],
        max_len=128)

    dataset = sentenizer.preprocess(dataset, use_cached=True)
    dataset = tokenizer.preprocess(dataset, use_cached=True)
    dataset = labelizer.preprocess(dataset, use_cached=True)
    dataset = collator.preprocess(dataset, use_cached=True)

    dataset.save(save_path)


if __name__ == '__main__':
    main()