import os
from pathlib import Path

from mlpipeline.datasets.nlp import Gazeta
from mlpipeline.processors.nlp.sentenizers import NatashaSentenizer
from mlpipeline.processors.nlp.tokenizers import RuBERTTokenizer
from mlpipeline.processors.nlp.collators import MaxLenSplitCollator, PaddingCollator
from mlpipeline.processors.numerator import Numerator


def main():
    home_dir = Path(os.getenv('HOME'))
    datasets_dir = home_dir / 'datasets'
    dataset_path = datasets_dir / 'Gazeta' / 'HuggingFaceLocal'
    save_path = datasets_dir / 'Gazeta' / 'PreprocessedForBert'
    # tokenizer name from Hugging Face Hub
    tokenizer_path = 'DeepPavlov/rubert-base-cased'

    dataset = Gazeta.load(dataset_path)

    numerator = Numerator(out_id_column='doc_id')
    sentenizer = NatashaSentenizer(text_column='text',
                                   doc_id_column='id',
                                   entities_column='entities',
                                   remove_columns=['summary', 'title', 'date',
                                                   'url'],
                                   out_text_column='text',
                                   out_start_column='start',
                                   out_stop_column='stop',
                                   out_doc_id_column='doc_id',
                                   out_entities_column='entities')
    tokenizer = RuBERTTokenizer(text_column='text',
                                pretrained_tokenizer_path=tokenizer_path,
                                out_tokens_column='tokens',
                                out_spans_column='spans',
                                out_tokens_ids_column='tokens_ids',
                                out_word_tokens_indices_column='out_word_tokens_indices_column',
                                out_attention_mask_column='attention_mask',
                                out_token_type_ids_column='token_type_ids')
    collator = MaxLenSplitCollator(
        collate_columns=['tokens', 'spans', 'tokens_ids', 'attention_mask',
                         'token_type_ids'],
        pk_columns=['doc_id', 'start'],
        unite_columns=['logits'],
        max_len=128
    )

    dataset = numerator.preprocess(dataset, use_cached=True)
    dataset = sentenizer.preprocess(dataset, use_cached=True)
    dataset = tokenizer.preprocess(dataset, use_cached=True)
    dataset = collator.preprocess(dataset, use_cached=True)

    dataset.save(save_path)


if __name__ == '__main__':
    main()
