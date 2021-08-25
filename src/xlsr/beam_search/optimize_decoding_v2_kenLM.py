import os
import yaml
import librosa

import torch
import torchaudio
import numpy as np
import optuna

from ctcdecode import CTCBeamDecoder
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from argparse import ArgumentParser, RawTextHelpFormatter
import preprocessing_text_japhug as prep_jap
import preprocessing_text_na as prep_na

DESCRIPTION = """
Train and optimize a KenLM language model from HuggingFace's provision.
Code updated from https://github.com/techiaith/docker-wav2vec2-xlsr-ft-cy/blob/main/train/python/train_kenlm.py
"""


# Preprocessing the datasets.
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(audio_path+batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 44_100, 16_000)
    return batch


def decode(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(logits)
    batch["pred_strings_with_lm"] = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]])

    return batch

def process(batch):
    batch['sentence'] = batch['sentence'].replace('|', ' ')
    return batch


def optimize_lm_objective(trial):
    global ctcdecoder

    alpha = trial.suggest_uniform('lm_alpha', 0, 6)
    beta = trial.suggest_uniform('lm_beta', 0, 5)

    try:
        ctcdecoder = CTCBeamDecoder(vocab,
                                    model_path=os.path.join(language_model_dir, "2grams.arpa"),
                                    alpha=alpha,
                                    beta=beta,
                                    cutoff_top_n=100,
                                    cutoff_prob=1.0,
                                    beam_width=100,
                                    num_processes=4,
                                    blank_id=processor.tokenizer.pad_token_id,
                                    log_probs_input=True
                                    )

        result = test_dataset.map(decode)
        result = result.map(process)
        print(result["sentence"][0])
        print(result["pred_strings_with_lm"][0])
        result_wer = wer.compute(predictions=result["pred_strings_with_lm"], references=result["sentence"])
        trial.report(result_wer, step=0)

    except Exception as e:
        print(e)
        raise

    finally:
        return result_wer


def optimize(kenlm_path, wav2vec_model_path, dataset, lang, clips):
    global processor
    global model
    global vocab
    global wer
    global test_dataset
    global language_model_dir
    global audio_path

    language_model_dir = kenlm_path
    audio_path = clips

    test_dataset = load_dataset('csv', data_files=[dataset], delimiter='\t')
    test_dataset = test_dataset['train']

    wer = load_metric("wer")

    processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_path)
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model_path)

    model.to("cuda")

    vocab = processor.tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))
    space_ix = vocab.index('|')
    vocab[space_ix] = ' '

    if lang == 'na':
        test_dataset = test_dataset.map(prep_na.final_text_words)

    elif lang == 'japhug':
        test_dataset = test_dataset.map(prep_jap.final_text_words)

    test_dataset = test_dataset.map(speech_file_to_array_fn)

    print("Beginning alpha and beta hyperparameter optimization")
    study = optuna.create_study()
    study.optimize(optimize_lm_objective, n_jobs=1, n_trials=100)

    lm_best = {'alpha': study.best_params['lm_alpha'], 'beta': study.best_params['lm_beta']}

    config_file_path = os.path.join(language_model_dir, "config_ctc.yaml")
    with open(config_file_path, 'w') as config_file:
        yaml.dump(lm_best, config_file)

    print('Best params saved to config file {}: alpha={}, beta={} with WER={}'.format(config_file_path,
                                                                                      study.best_params['lm_alpha'],
                                                                                      study.best_params['lm_beta'],
                                                                                      study.best_value))


def main(args):
    optimize(args.path_lm, args.model, args.test_data, args.lang, args.path_clips)


if __name__ == "__main__":
    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)

    parser.add_argument("--model", required=True,
                        help="acoustic model to be used for optimizing")
    parser.add_argument("--test_data", required=True,
                        help="dataset")
    parser.add_argument("--lang", required=True, choices={"japhug", "na"},
                        help="Lang of the corpus")
    parser.add_argument("--path_clips", required=True,
                        help="path to the audio clips from the chosen corpus.")
    parser.add_argument("--path_lm", required=True,
                        help="path to the kenLM language model.")
    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(args)
