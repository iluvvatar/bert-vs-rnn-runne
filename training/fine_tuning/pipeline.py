# =============================================================================
#
# =============================================================================
import torch
from pathlib import Path
import os

from torch.optim import Adam
from torch.utils.data import DataLoader

from mlpipeline.datasets import RuNNE
from mlpipeline.datasets.nlp.units import Entity
from mlpipeline.datasets.nlp.samplers import UnbalancedEntitiesSampler

from mlpipeline.processors.nlp.prediction_postprocessors import Viterbi
from mlpipeline.processors.nlp.labelizers import BILOULabelizer
from mlpipeline.processors.nlp.collators import PaddingCollator

from mlpipeline.trainers.utils.losses import CrossEntropyLoss, DiceLoss, LossList
from mlpipeline.trainers.utils.metrics import F1MacroScoreNER
from mlpipeline.trainers.utils.callbacks import CheckpointCallback, LoggingCallback
from mlpipeline.trainers.utils.lr_schedulers import PlateauScheduler
from mlpipeline.trainers import NerTrainer


few_shot_ent_types = ["DISEASE", "PENALTY", "WORK_OF_ART"]


def fine_tuning_pipeline(model, pretrained_gazeta_model_path, device, experiment_name, checkpoint_name=None):
    epochs = 10
    batch_size = 512

    home_dir = Path(os.getenv('HOME'))
    datasets_dir = home_dir / 'datasets'
    output_dir = Path(os.getenv('SCRATCH_DIR')) / 'models' / model.name / 'fine_tuning_runne' / experiment_name

    runne_path = datasets_dir / 'RuNNE' / 'Preprocessed'

    runne = RuNNE.load(runne_path)
    train_ds = runne['train']
    val_ds = runne['test']
    sampler = UnbalancedEntitiesSampler(train_ds,
                                        entities_column='entities',
                                        entities_deserialize_fn=Entity.from_str,
                                        tokens_spans_column='spans',
                                        entity_types_shares='log',
                                        size=100_000)
    collator = PaddingCollator(
        collate_columns=['tokens_ids', 'attention_mask', 'labels_ids'],
        pad_value=0,
        padding_type='longest')
    train_loader = DataLoader(train_ds,
                              sampler=sampler,
                              batch_size=batch_size,
                              collate_fn=collator.collate)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            collate_fn=collator.collate)

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
                               out_labels_column='labels',
                               out_labels_ids_column='labels_ids',
                               predicted_labels_ids_column='predicted_labels_ids',
                               out_predicted_entities_column='predicted_entities',
                               entity_types=runne.entity_types,
                               entities_deserialize_fn=Entity.from_str,
                               entities_serialize_fn=Entity.to_str)

    model.load_state_dict(torch.load(pretrained_gazeta_model_path, map_location=torch.device(device)))
    model.freeze_embeddings(requires_grad=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable parameters:', total_params)

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = PlateauScheduler(optimizer,
                                 mode='max',
                                 factor=0.5,
                                 min_lr=1e-4,
                                 patience=0,
                                 threshold=0,
                                 threshold_mode='rel')
    loss_fn = LossList([CrossEntropyLoss(label_smoothing=0.01),
                        DiceLoss(weights=[0, 1, 1, 1, 1])],
                       ratios=[0.2, 0.8])
    loss_fn = LossList([loss_fn for ent_type in runne.entity_types]).to(device)

    callbacks = [CheckpointCallback(output_dir / 'checkpoints', 10),
                 LoggingCallback()]
                 # EarlyStoppingCallback(patience=40)
    metrics = [
        F1MacroScoreNER(entity_types=runne.entity_types,
                        # few_shot_entity_types=few_shot_ent_types,
                        entities_column='entities',
                        predicted_entities_column='predicted_entities',
                        entities_deserialize_fn=Entity.from_str,
                        name='f1-macro')
    ]
    key_metric_name = 'f1-macro'

    trainer = NerTrainer(model=model,
                         optimizer=optimizer,
                         loss_fn=loss_fn,
                         lr_scheduler=scheduler,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         tokens_ids_column='tokens_ids',
                         attention_mask_column='attention_mask',
                         labels_ids_column='labels_ids',
                         logits_column='logits',
                         callbacks=callbacks,
                         metrics=metrics,
                         key_metric_name=key_metric_name,
                         viterbi_decoder=viterbi,
                         labelizer=labelizer,
                         device=device,
                         output_dir=output_dir,
                         verbose=False)

    if checkpoint_name is not None:
        checkpoint_path = output_dir / 'checkpoints' / checkpoint_name
        trainer.load_checkpoint(checkpoint_path)

    trainer.train(epochs)
