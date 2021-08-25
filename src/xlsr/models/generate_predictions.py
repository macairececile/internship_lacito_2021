# -*- coding: utf-8 -*-
# ----------- Libraries ----------- #
import argparse
import difflib
import itertools
from os import listdir
from os.path import isfile, join

import librosa
import torchaudio
from datasets import load_dataset
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import torch
import preprocessing_text_na as prep
import preprocessing_audio_na as prep_audio
import preprocessing_text_japhug as prep_jap
import preprocessing_audio_japhug as prep_audio_jap
import numpy as np
import csv
import json


# ----------- Functions ----------- #
def get_files_from_directory(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files


def load_model(model_path):
    # Call the fine-tuned model
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to("cuda")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    return model, processor, tokenizer


def load_data(file):
    na_test = load_dataset('csv', data_files=[file], delimiter='\t')
    na_test = na_test['train']
    return na_test


def pipeline(model_dir, test_file, language):
    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
                len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    # load model
    model, processor, tokenizer = load_model(model_dir)
    # load the dataset
    data = load_data(test_file)

    if language == "na":
        na_test = data.map(prep.final_text_words)
        na_test_ref = na_test.map(prep_audio.speech_file_to_array_fn, remove_columns=na_test.column_names)
        na_test_ref = na_test_ref.map(prep_audio.resample, num_proc=4)
        na_test_ref = na_test_ref.map(prepare_dataset, remove_columns=na_test_ref.column_names, batch_size=8,
                                      num_proc=4,
                                      batched=True)
        return model, processor, tokenizer, na_test, na_test_ref
    elif language == "japhug":
        na_test = data.map(prep_jap.final_text_words)
        na_test_ref = na_test.map(prep_audio_jap.speech_file_to_array_fn, remove_columns=na_test.column_names)
        na_test_ref = na_test_ref.map(prep_audio_jap.resample, num_proc=4)
        na_test_ref = na_test_ref.map(prepare_dataset, remove_columns=na_test_ref.column_names, batch_size=8,
                                      num_proc=4,
                                      batched=True)
        return model, processor, tokenizer, na_test, na_test_ref


def save_predictions(arguments):
    # Preprocessing the data
    model, processor, tokenizer, na_test, na_test_ref = pipeline(arguments.model_dir, arguments.test_tsv,
                                                                 arguments.lang)
    ref = []
    pred = []
    for i in range(len(na_test_ref)):
        input_dict = processor(na_test_ref["input_values"][i], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to("cuda")).logits
        torch.save(logits, '/data/user/m/cmacaire/xlsr53/japhug/model_250/tensors_japhug/logits_' + str(i) + '.pt')
        pred_ids = torch.argmax(logits, dim=-1)[0]
        pred.append(processor.decode(pred_ids))
        ref.append(na_test['sentence'][i])
        print(i)
        # print(processor.decode(pred_ids))
    # store the results in a CSV file
    # df_ref = pd.DataFrame({'Reference': ref})
    # df_ref.to_csv('refs_na.csv', index=False, sep='\t')
    df_results = pd.DataFrame({'Reference': ref,
                               'Prediction': pred})
    df_results.to_csv(arguments.model_dir + 'results.csv', index=False, sep='\t')


def show_predictions(arguments):
    # Preprocessing the data
    model, processor, tokenizer, na_test, na_test_ref = pipeline(arguments.model_dir, arguments.test_tsv,
                                                                 arguments.lang)
    if arguments.num_pred < len(na_test_ref):
        for i in range(arguments.num_pred):
            input_dict = processor(na_test_ref["input_values"][i], return_tensors="pt", padding=True,
                                   sampling_rate=16000)
            logits = model(input_dict.input_values.to("cuda")).logits
            pred_ids = torch.argmax(logits, dim=-1)[0]
            print("Prediction:")
            print(processor.decode(pred_ids))
            print("Reference:")
            print(na_test['sentence'][i])


def predict_audio(arguments):
    # Preprocessing the data
    model = Wav2Vec2ForCTC.from_pretrained(arguments.model_dir).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(arguments.model_dir)

    predictions = []

    files = get_files_from_directory(arguments.input_dir)
    
    for i in files:
        speech_array, sampling_rate = torchaudio.load(arguments.input_dir+i)
        speech = speech_array[0].numpy()
        speech = librosa.resample(np.asarray(speech), 44_100, 16_000)
        input_dict = processor(speech, return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to("cuda")).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        print(pred_ids)
        # print("Prediction on an unseen audio:")
        predictions.append(processor.decode(pred_ids))
        # print(processor.decode(pred_ids))
        
    df_results = pd.DataFrame({'File': files,
                                   'Prediction': predictions})
    df_results.to_csv(arguments.input_dir + 'results.csv', index=False, sep='\t')


# ----------- Arguments ----------- #
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

save_pred = subparsers.add_parser("save_predict",
                                  help="Generate predictions from fine-tuned model and store them in csv file.")
save_pred.add_argument('--test_tsv', type=str, required=True,
                       help="Test .tsv file.")
save_pred.add_argument('--model_dir', type=str, required=True,
                       help="Directory where the fine-tuned model is stored.")
save_pred.add_argument('--lang', type=str, required=True, choices={"japhug", "na"},
                       help="Language of the corpus.")
save_pred.set_defaults(func=save_predictions)

gen_pred = subparsers.add_parser("generate_predict",
                                 help="Generate predictions from fine-tuned model and print them.")
gen_pred.add_argument('--test_tsv', type=str, required=True,
                      help="Test .tsv file.")
gen_pred.add_argument('--model_dir', type=str, required=True,
                      help="Directory where the fine-tuned model is stored.")
gen_pred.add_argument('--num_pred', type=int, required=False,
                      help="Number of predictions to show")
gen_pred.add_argument('--lang', type=str, required=True, choices={"japhug", "na"},
                      help="Language of the corpus.")
gen_pred.set_defaults(func=show_predictions)

gen_audio = subparsers.add_parser("predict_audios",
                                  help="Generate predictions from fine-tuned model for wav input files.")
gen_audio.add_argument('--input_dir', type=str, required=True,
                       help="Directory to wav files to transcribe.")
gen_audio.add_argument('--model_dir', type=str, required=True,
                       help="Directory where the fine-tuned model is stored.")
gen_audio.set_defaults(func=predict_audio)

arguments = parser.parse_args()
arguments.func(arguments)
