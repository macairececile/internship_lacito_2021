# -*- coding: utf-8 -*-
# ----------- Libraries ----------- #
import argparse
from datasets import load_dataset
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import preprocessing_text_na as prep
import preprocessing_audio_na as prep_audio
import preprocessing_text_japhug as prep_jap
import preprocessing_audio_japhug as prep_audio_jap
import numpy as np


# ----------- Functions ----------- #
def load_model(model_path):
    # Call the fine-tuned model
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    return model, processor


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
    model, processor = load_model(model_dir)
    # load the dataset
    data = load_data(test_file)

    if language == "na":
        na_test = data.map(prep.final_text_words)
        na_test_ref = na_test.map(prep_audio.speech_file_to_array_fn, remove_columns=na_test.column_names)
        na_test_ref = na_test_ref.map(prep_audio.resample, num_proc=4)
        na_test_ref = na_test_ref.map(prepare_dataset, remove_columns=na_test_ref.column_names, batch_size=8,
                                      num_proc=4,
                                      batched=True)
        return model, processor, na_test, na_test_ref
    elif language == "japhug":
        na_test = data.map(prep_jap.final_text_words)
        na_test_ref = na_test.map(prep_audio_jap.speech_file_to_array_fn, remove_columns=na_test.column_names)
        na_test_ref = na_test_ref.map(prep_audio_jap.resample, num_proc=4)
        na_test_ref = na_test_ref.map(prepare_dataset, remove_columns=na_test_ref.column_names, batch_size=8,
                                      num_proc=4,
                                      batched=True)
        return model, processor, na_test, na_test_ref


def save_predictions(arguments):
    # Preprocessing the data
    model, processor, na_test, na_test_ref = pipeline(arguments.model_dir, arguments.test_tsv, arguments.lang)
    ref = []
    pred = []
    for i in range(len(na_test_ref)):
        input_dict = processor(na_test_ref["input_values"][i], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to("cuda")).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        pred.append(processor.decode(pred_ids))
        ref.append(na_test['sentence'][i])
        # print(processor.decode(pred_ids))
    # store the results in a CSV file
    df_results = pd.DataFrame({'Reference': ref,
                               'Prediction': pred})
    df_results.to_csv(arguments.model_dir + 'results.csv', index=False, sep='\t')


def show_predictions(arguments):
    # Preprocessing the data
    model, processor, na_test, na_test_ref = pipeline(arguments.model_dir, arguments.test_tsv, arguments.lang)
    if arguments.num_pred < len(na_test_ref):
        for i in range(arguments.num_pred):
            input_dict = processor(na_test_ref["input_values"][i], return_tensors="pt", padding=True,
                                   sampling_rate=16000)
            logits = model(input_dict.input_values.to("cuda")).logits

            # TEST TO DELETE LATER ------------------------------------------------ #
            # print(processor.decode(torch.argmax(logits, dim=-1)[0]))
            # topK_val, topK_ind = torch.topk(logits[0, -1, :], 10, dim=-1)
            # print(topK_ind)
            # print(processor.decode(topK_ind[0][i]))
            dict_top = {}
            for j in range(len(logits[0])):
                topK_val, topK_ind = torch.topk(logits[0][j], 10, dim=-1)
                for k, m in enumerate(topK_ind.cpu().numpy()):
                    if k in dict_top.keys():
                        dict_top[k] = np.append(dict_top[k], m)
                    else:
                        dict_top[k] = m

            pred = []
            for v in dict_top.values():
                print(v)
                pred.append(processor.decode(torch.from_numpy(v)))

            df_results = pd.DataFrame({'Prediction': pred})
            df_results.to_csv(arguments.model_dir + 'results_n_best_hyp.csv', index=False, sep='\t')
            # ------------------------------------------------ #

            pred_ids = torch.argmax(logits, dim=-1)[0]
            print("Prediction:")
            print(processor.decode(pred_ids))
            print("Reference:")
            print(na_test['sentence'][i])


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

arguments = parser.parse_args()
arguments.func(arguments)
