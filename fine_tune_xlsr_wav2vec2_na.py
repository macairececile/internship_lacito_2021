# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

train = subparsers.add_parser("train",
                              help="Train XLSR-53 model.")
train.add_argument('--train_tsv', type=str, required=True,
                   help="Train .tsv file.")
train.add_argument('--test_tsv', type=str, required=True,
                   help="Test .tsv file.")
train.add_argument('--output_dir', type=str, required=True,
                   help="Output directory to store the fine-tuned model.")
arguments = parser.parse_args()

# ------------ Libraries ------------ #
from datasets import load_dataset, load_metric
import pandas as pd
from IPython.display import display, HTML
import json
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import random
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import preprocessing_text_na as prep
import preprocessing_audio_na as prep_audio

# ------------ Load dataset ------------ #
na_train = load_dataset('csv', data_files=[arguments.train_tsv], delimiter='\t')
na_test = load_dataset('csv', data_files=[arguments.test_tsv], delimiter='\t')

na_train = na_train['train']
na_test = na_test['train']


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))


# show_random_elements(na_train, num_examples=5)

# ----------- Preprocessing ----------- #
# cut by phonemes
# na_train = na_train.map(final_text_phonemes)
# na_test = na_test.map(final_text_phonemes)

# cut by words
na_train = na_train.map(prep.final_text_words)
na_test = na_test.map(prep.final_text_words)


# show_random_elements(na_train.remove_columns(["path"]), num_examples=5)

# ------------ Vocabulary ------------ #
def extract_all_chars(batch):
    all_text = "".join(batch["sentence"])
    vocab = list(set(all_text))
    # voc = []
    # for i in batch['phonemes']:
    #    voc.append(list(set(i)))
    # voc = [item for l in voc for item in l]
    # vocab = list(set(voc))
    return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = na_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                           remove_columns=na_train.column_names)
vocab_test = na_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                          remove_columns=na_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict['c'] = len(vocab_dict)
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

with open(arguments.output_dir + 'vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# ------------ Tokenizer, Feature extractor & Processor ------------ #
tokenizer = Wav2Vec2CTCTokenizer(arguments.output_dir + "vocab.json",
                                 unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

processor.save_pretrained(arguments.output_dir)

# ------------ Preprocessing dataset audio ------------ #
na_train = na_train.map(prep_audio.speech_file_to_array_fn, remove_columns=na_train.column_names)
na_test = na_test.map(prep_audio.speech_file_to_array_fn, remove_columns=na_test.column_names)
na_train = na_train.map(prep_audio.resample, num_proc=4)
na_test = na_test.map(prep_audio.resample, num_proc=4)
na_train = na_train.map(prep_audio.prepare_dataset, remove_columns=na_train.column_names, batch_size=8, num_proc=4, batched=True)
na_test = na_test.map(prep_audio.prepare_dataset, remove_columns=na_test.column_names, batch_size=8, num_proc=4, batched=True)


# ------------ Dataclass ------------ #
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# metrics to choose
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    # cer = cer_metric.compute(predictions=pred_str, references=label_str)
    # return {"cer": cer}
    return {"wer": wer}


# ------------ Definition of the model, training args ------------ #
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.075,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_extractor()

training_args = TrainingArguments(
    output_dir=arguments.output_dir,
    # output_dir="./wav2vec2-large-xlsr-turkish-demo",
    logging_dir=arguments.output_dir,
    group_by_length=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=60,
    fp16=True,
    save_steps=100,
    eval_steps=50,
    logging_steps=50,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=na_train,
    eval_dataset=na_test,
    tokenizer=processor.feature_extractor,
)

trainer.train()
