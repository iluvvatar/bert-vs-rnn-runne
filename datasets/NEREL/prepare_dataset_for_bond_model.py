import os
from pathlib import Path

from mlpipeline.datasets.nlp import NEREL
from mlpipeline.processors.nlp.sentenizers import NatashaSentenizer
from mlpipeline.processors.nlp.tokenizers import RuBERTTokenizer
from mlpipeline.processors.nlp.labelizers import BILOULabelizer
from mlpipeline.processors.nlp.collators import MaxLenSplitCollator

from mlpipeline.utils import PathLike


def main(dataset_folder_path: PathLike,
         save_to_folder: PathLike,
         tokenizer_path: PathLike):
    dataset = NEREL.load_from_disk(dataset_folder_path)

    sentenizer = NatashaSentenizer(text_column='text',
                                   entities_column='entities',
                                   doc_id_column='id',
                                   remove_columns=['relations', 'links', 'id'],
                                   out_text_column='text',
                                   out_start_column='start',
                                   out_stop_column='stop',
                                   out_doc_id_column='doc_id',
                                   out_entities_column='entities')
    tokenizer = RuBERTTokenizer(text_column='text',
                                pretrained_tokenizer_path=tokenizer_path)
    labelizer = BILOULabelizer(text_column='text',
                               tokens_column='tokens',
                               tokens_spans_column='spans',
                               entities_column='entities',
                               out_labels_column='labels',
                               out_labels_ids_column='labels_ids',
                               entity_types=dataset.entity_types)

    collator = MaxLenSplitCollator(
        collate_columns=['tokens', 'spans', 'tokens_ids', 'attention_mask',
                         'token_type_ids', 'labels', 'labels_ids'],
        pk_columns=['doc_id', 'start'],
        unite_columns=[],
        max_len=128
    )
    dataset = sentenizer.preprocess(dataset, use_cached=False)
    dataset = tokenizer.preprocess(dataset, use_cached=False)
    dataset = labelizer.preprocess(dataset, use_cached=False)
    dataset = collator.preprocess(dataset, use_cached=False)

    dataset.save(save_to_folder)


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    datasets_dir = home_dir / 'datasets'
    models_dir = home_dir / 'models'
    dataset_path = datasets_dir / 'NEREL' / 'HuggingFaceLocal'
    save_to_path = datasets_dir / 'NEREL' / 'PreprocessedForBondModel'
    tokenizer_path = models_dir / 'tokenizers' / 'DeepPavlov-rubert-base-cased'
    main(dataset_path, save_to_path, tokenizer_path)
