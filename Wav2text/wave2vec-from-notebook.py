# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 20:32:47 2021

@author: Antoine Misery
"""
#preproc & general modules
from datasets import load_dataset, ClassLabel, load_metric, Audio, Dataset
import random
import pandas as pd
from IPython.display import display, HTML
import librosa
import numpy as np
from ast import literal_eval
import json
import re

#training modules
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

common_voice_train = load_dataset("common_voice", "br", split="train+validation")
common_voice_test = load_dataset("common_voice", "br", split="test")
#common_voice_train = load_dataset("br", data_files = ["train.tsv","validated.tsv"], split = "train+validation")
#common_voice_test = load_dataset("br", data_files = "test.tsv", split = "test")
#common_voice_train = pd.read_csv("br/train.tsv", sep = "\t")[:10]
#common_voice_test = pd.read_csv("br/test.tsv", sep = "\t")[:10]   

#common_voice_train = Dataset.from_pandas(common_voice_train)
#common_voice_test = Dataset.from_pandas(common_voice_test)
common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
"""
train_dict = json.load(open("train.json"))
test_dict = json.load(open("test.json"))
audio_train = []
audio_test = []
for clip in common_voice_train["path"]:
    audio_train.append(np.array(train_dict[clip]))
for clip in common_voice_test["path"]:
    audio_test.append(np.array(test_dict[clip]))
    
common_voice_train = common_voice_train.add_column("audio", np.array(audio_train))
common_voice_test = common_voice_test.add_column("audio", np.array(audio_test))
"""
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

#Remove unwanted characters

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch
print('removing special characters')
common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
    
#Create audio data with librosa
"""
audio_train = []
audio_test = []
for clip in common_voice_train['path']:
    print(clip)
    y,sr = librosa.load("br/clips/" + clip, sr = 16000)
    print(audio_train)
    audio_train.append({'array':y , 'path':clip,'sampling_rate':16000})
for clip in common_voice_test['path']:
    y,sr = librosa.load("br/clips/" + clip, sr = 16000)
    audio_test.append({'array':y , 'path':clip,'sampling_rate':16000})

common_voice_train.add_column('audio', audio_train)
common_voice_test.add_column('audio', audio_test)
print(audio_train)
"""

#tokenizer
print('loading tokenizer')
from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

print("loading feature extractor")
#feature extractor
from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

#Besides normalization, one can include advanced types of preprocessing, such as Log-Mel feature extraction.
#processor, which combines tokenizer & feature extractor
print("building processor")
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

#checking that the data is correctly loaded and sampled
rand_int = random.randint(0, len(common_voice_train)-1)

print("Target text:", common_voice_train["sentence"][rand_int])
#print(common_voice_train["audio"][rand_int])
#print("Sampling rate:", common_voice_train["audio"][rand_int]["Sampling rate"])

def prepare_dataset(batch):
    #Loading and resampling the audio data
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    #proper way to use processor
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

print("preparing dataset...")
common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)

# TRAINING

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
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

#Model loading
from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m", 
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

model.freeze_feature_extractor()

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir='./',
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  gradient_checkpointing=True,
  save_steps=400,
  eval_steps=400,
  logging_steps=400,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=2,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)
#trainer.train()
"""
input_dict = processor(common_voice_test[9]["input_values"], return_tensors="pt", padding=True)
print(input_dict)
logits = model(input_dict.input_values).logits
print(logits)
pred_ids = torch.argmax(logits, dim=-1)[0]
print(pred_ids)
print(processor.decode(pred_ids))
"""
test_dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
audio_sample = test_dataset[2]
with torch.no_grad():
    inputs = processor(audio_sample["audio"]["array"], return_tensors="pt", padding=True)
logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print(transcription[0].lower())
print(transcription)
