import codecs
import json
import os
import re
from typing import List, Tuple

import numpy as np
from scipy.stats import hmean
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import BertConfig, TFBertModel, BertTokenizer

from .utils import AttentionMaskLayer, MaskCalculator
from .utils import generate_random_seed


def get_nn_output_name(target_name: str) -> str:
    return target_name.title().replace('-', '').replace(':', '')


def load_ner(path: str,
             tokenizer_path: str,
             bert_config_path: str) -> Tuple[tf.keras.Model,
                                             BertTokenizer,
                                             int,
                                             List[str]]:
    config_name = os.path.join(path, 'ner.json')
    weights_name = os.path.join(path, 'ner.h5')
    with codecs.open(config_name, mode='r', encoding='utf-8') as fp:
        config_data = json.load(fp)
    base_name = config_data["base_name"]
    bert_config = BertConfig.from_pretrained(bert_config_path)
    output_embedding_size = bert_config.hidden_size
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    word_ids = tf.keras.layers.Input(
        shape=(config_data['max_sent_len'],),
        dtype=tf.int32,
        name=f'base_word_ids_{base_name}'
    )
    attention_mask = AttentionMaskLayer(
        pad_token_id=bert_config.pad_token_id,
        name=f'base_attention_mask_{base_name}',
        trainable=False
    )(word_ids)
    transformer_layer = TFBertModel(
        config=bert_config,
        name=f'BertNLU_{base_name}'
    )
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    output_mask = MaskCalculator(
        output_dim=output_embedding_size,
        pad_token_id=bert_config.pad_token_id,
        trainable=False,
        name=f'MaskCalculator_{base_name}'
    )(word_ids)
    masked_sequence_output = tf.keras.layers.Multiply(
        name='MaskedOutput_'
    )([output_mask, sequence_output])
    masked_sequence_output = tf.keras.layers.Masking(
        name='MaskedEmdOutput', mask_value=0.0
    )(masked_sequence_output)
    outputs = []
    for cur_entity in config_data['named_entities']:
        new_layer_name = get_nn_output_name(cur_entity) + f'_{base_name}'
        new_dropout_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(
                rate=0.5,
                seed=generate_random_seed(),
                name=new_layer_name + '_dropout_'
            ),
            name=new_layer_name + '_dropout'
        )(masked_sequence_output)
        new_output_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                units=5,
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_normal(
                    seed=generate_random_seed()
                ),
                bias_initializer=tf.keras.initializers.zeros(),
                name=new_layer_name + '_'
            ),
            name=new_layer_name
        )(new_dropout_layer)
        outputs.append(new_output_layer)
    ner_model = tf.keras.Model(
        word_ids,
        outputs,
        name=f'NamedEntityRecognizer_{base_name}'
    )
    ner_model.build(input_shape=(None, config_data['max_sent_len']))
    ner_model.load_weights(weights_name)
    return ner_model, tokenizer, config_data['max_sent_len'], \
           config_data['named_entities']
