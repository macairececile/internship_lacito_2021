# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

save_pred = subparsers.add_parser("save_predict",
                                  help="Generate predictions from fine-tuned model and store them in csv file.")
save_pred.add_argument('--test_tsv', type=str, required=True,
                       help="Test .tsv file.")
save_pred.add_argument('--model_dir', type=str, required=True,
                       help="Directory where the fine-tuned model is stored.")

gen_pred = subparsers.add_parser("generate_predict",
                                 help="Generate predictions from fine-tuned model and print them.")
gen_pred.add_argument('--test_tsv', type=str, required=True,
                      help="Test .tsv file.")
gen_pred.add_argument('--model_dir', type=str, required=True,
                      help="Directory where the fine-tuned model is stored.")
gen_pred.add_argument('--num_pred', type=int, required=False,
                      help="Number of predictions to show")
arguments = parser.parse_args()

# ----------- Libraries ----------- #
from datasets import load_dataset
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import preprocessing_text_na as prep

# ----------- Generate Predictions ----------- #
# Call the fine-tuned model
model_path = arguments.model_dir
model = Wav2Vec2ForCTC.from_pretrained(model_path).to("cuda")
processor = Wav2Vec2Processor.from_pretrained(model_path)

# Load the test data
na_test_ref = load_dataset('csv', data_files=[arguments.test_tsv], delimiter='\t')
na_test_ref = na_test_ref['train']
na_test_ref = na_test_ref.map(prep.final_text_words)

# Predict the transcription from the test data
if gen_pred:
    if arguments.num_pred < len(na_test_ref):
        for i in range(arguments.num_pred):
            input_dict = processor(na_test_ref["input_values"][i], return_tensors="pt", padding=True,
                                   sampling_rate=16000)
            logits = model(input_dict.input_values.to("cuda")).logits
            pred_ids = torch.argmax(logits, dim=-1)[0]
            print("Prediction:")
            print(processor.decode(pred_ids))
            print("Reference:")
            print(na_test_ref['sentence'][i])

if save_pred:
    ref = []
    pred = []
    for i in range(len(na_test_ref)):
        input_dict = processor(na_test_ref["input_values"][i], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to("cuda")).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        pred.append(processor.decode(pred_ids))
        ref.append(na_test_ref['sentence'][i])
        # print(processor.decode(pred_ids))
    # store the results in a CSV file
    df_results = pd.DataFrame({'Reference': ref,
                               'Prediction': pred})
    df_results.to_csv(arguments.model_dir + 'results.csv', index=False, sep='\t')