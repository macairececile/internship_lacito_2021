# -*- coding: utf-8 -*-
# ----------- Libraries ----------- #
import argparse
import difflib
from datasets import load_dataset, load_metric
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import torch
import preprocessing_text_na as prep
import preprocessing_audio_na as prep_audio
import preprocessing_text_japhug as prep_jap
import preprocessing_audio_japhug as prep_audio_jap
import numpy as np
import json
import ctcdecodebis
from ctcdecode import CTCBeamDecoder
import operator

cer_metric = load_metric('cer')
wer_metric = load_metric('wer')


# ----------- Load the data, the model, tokenizer, processor, process the data ----------- #
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


# ----------- Top k hypotheses ----------- #
def top_k_hypotheses(logits, k):
    logits = logits.cpu().detach().numpy().tolist()[0]  # tensor to list
    sequences = [[list(), 0.0, list()]]
    # walk over each step in sequence
    for row in logits:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score, scores = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - row[j], scores + [row[j]]]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences


# Print the different tokens from the best paths for k paths
def show_replace_tokens(tokenizer, data_dict, k):
    for el in range(1, k):
        matcher = difflib.SequenceMatcher(None, data_dict[0], data_dict[el])
        for tag, i1, i2, j1, j2 in reversed(matcher.get_opcodes()):
            if tag == 'replace':
                print('{:7}   [{}:{}] --> [{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2,
                                                                           data_dict[0][i1:i2],
                                                                           data_dict[el][j1:j2]))
                print(tokenizer.convert_ids_to_tokens(data_dict[0][i1:i2]))
                print(tokenizer.convert_ids_to_tokens(data_dict[el][j1:j2]))
            if tag == 'insert':
                print('{:7}   [{}:{}] --> [{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2,
                                                                           data_dict[0][i1:i2],
                                                                           data_dict[el][j1:j2]))
                print(tokenizer.convert_ids_to_tokens(data_dict[0][i1:i2]))
                print(tokenizer.convert_ids_to_tokens(data_dict[el][j1:j2]))


# Calculate the oracle scores for the k hypothesis
def oracle_score_computation(metric, ref, preds, all_score, all_min_preds):
    scores = list(map(lambda a: metric.compute(predictions=[a], references=[ref]), preds))
    final_scores = zip(preds, scores)
    max_score = min(final_scores, key=operator.itemgetter(1))
    all_score.append(max_score)
    all_min_preds.append(max_score[0])
    return all_min_preds


def oracle_score(csv_file):
    df = pd.read_csv(csv_file, sep='\t')
    all_score_cer = []
    all_score_wer = []
    ids = list(set(df['Id_sentence']))
    all_min_refs = []
    all_min_preds_cer = []
    all_min_preds_wer = []
    for i in ids:
        ref = df.loc[df['Id_sentence'] == i]['Reference'].tolist()[0].replace('|', ' ')
        preds = df.loc[df['Id_sentence'] == i]['Prediction'].tolist()
        all_min_preds_cer = oracle_score_computation(cer_metric, ref, preds, all_score_cer, all_min_preds_cer)
        all_min_preds_wer = oracle_score_computation(wer_metric, ref, preds, all_score_wer, all_min_preds_wer)
        all_min_refs.append(ref)
    oracle_score_cer = cer_metric.compute(predictions=all_min_preds_cer, references=all_min_refs)
    oracle_score_wer = wer_metric.compute(predictions=all_min_preds_wer, references=all_min_refs)
    return oracle_score_wer, oracle_score_cer


# ----------- Beam search decoding ----------- #
# decoding with https://github.com/ynop/py-ctc-decode
def beam_search_decoder_lm(tokenizer, logits, lm=None):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1].replace('|', ' ') if x[1] not in tokenizer.all_special_tokens else "_" for x in sort_vocab]
    print(vocab)
    decoder = ctcdecodebis.BeamSearchDecoder(
        vocab,
        num_workers=4,
        beam_width=100,
        scorers=[lm],
        cutoff_prob=np.log(0.000001),
        cutoff_top_n=100
    )
    prediction_lm = decoder.decode(logits.cpu().detach().numpy()[0])
    return prediction_lm


# decoding with https://github.com/parlance/ctcdecode
def beam_search_decoder_lm_v2(processor, tokenizer, logits, lm, alpha, beta):
    vocab = tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))
    space_ix = vocab.index('|')
    vocab[space_ix] = ' '

    ctcdecoder = CTCBeamDecoder(vocab,
                                model_path=lm,
                                alpha=alpha,
                                beta=beta,
                                cutoff_top_n=100,
                                cutoff_prob=1.0,
                                beam_width=100,
                                num_processes=4,
                                blank_id=processor.tokenizer.pad_token_id,
                                log_probs_input=True
                                )

    beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(logits)

    # beam_results - Shape: BATCHSIZE x N_BEAMS X N_TIMESTEPS A batch containing the series
    # of characters (these are ints, you still need to decode them back to your text) representing
    # results from a given beam search. Note that the beams are almost always shorter than the
    # total number of timesteps, and the additional data is non-sensical, so to see the top beam
    # (as int labels) from the first item in the batch, you need to run beam_results[0][0][:out_len[0][0]].
    beam_string = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]])

    # timesteps : BATCHSIZE x N_BEAMS : the timestep at which the nth output character has peak probability.
    # Can be used as alignment between the audio and the transcript.
    alignment = list()
    for i in range(0, out_lens[0][0]):
        alignment.append([beam_string[i], int(timesteps[0][0][i])])
    return beam_string


def decoding_top_k(arguments):
    # Preprocessing the data
    model, processor, tokenizer, na_test, na_test_ref = pipeline(arguments.model_dir, arguments.test_tsv,
                                                                 arguments.lang)
    decode_k_best_hyp = []
    index = []
    refs_k_best_hyp = []
    for i in range(len(na_test)):
        # ------ load the tensor ------ #
        input_dict = processor(na_test_ref["input_values"][i], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to("cuda")).logits
        # ------ beam search algorithm  ------ #
        k_best_hyp = top_k_hypotheses(logits, arguments.k_hyp)
        data_json = {'predicted_ids': [], 'scores': [], 'names': [], 'total_score': []}
        dict_seq = {}
        for a, b in enumerate(k_best_hyp):
            dict_seq[a] = b[0]
            data_json['predicted_ids'].append(b[0])
            data_json['scores'].append(b[2])
            data_json['names'].append(tokenizer.convert_ids_to_tokens(b[0]))
            data_json['total_score'].append(b[1])
            index.append(i)
            decode_k_best_hyp.append(processor.decode(torch.from_numpy(np.array(b[0]))))
            refs_k_best_hyp.append(na_test['sentence'][i])
        # ------ save the k best paths in a json file for visualization ------ #
        with open(arguments.json_path + 'data_json_' + str(i) + '.json', 'w') as outfile:
            json.dump(data_json, outfile)
        # ------ Show the tokens changed in the k best hyp ------ #
        show_replace_tokens(tokenizer, dict_seq, arguments.k_hyp)
    # ------ save the k best paths in a csv file ------ #
    df_results = pd.DataFrame({'Id_sentence': index, 'Reference': refs_k_best_hyp, 'Prediction': decode_k_best_hyp})
    df_results.to_csv(arguments.model_dir + 'results_' + str(arguments.k_hyp) + '_best_hyp_' + arguments.lang + '.csv',
                      index=False, sep='\t')
    # ------ calculate the oracle score ------ #
    wer, cer = oracle_score(
        arguments.model_dir + 'results_' + str(arguments.k_hyp) + '_best_hyp_' + arguments.lang + '.csv')
    print("WER oracle score : ", wer)
    print("CER oracle score : ", cer)


def decoding_lm(arguments):
    model, processor, tokenizer, na_test, na_test_ref = pipeline(arguments.model_dir, arguments.test_tsv,
                                                                 arguments.lang)
    preds_lm = []
    # LM parameters
    alpha = arguments.alpha  # LM Weight
    beta = arguments.beta  # LM Usage Reward
    word_lm_scorer = ctcdecodebis.WordKenLMScorer(arguments.lm, alpha, beta)
    refs = []
    for i in range(len(na_test)):
        # ------ load the tensor ------ #
        input_dict = processor(na_test_ref["input_values"][i], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to("cuda")).logits
        preds_lm.append(beam_search_decoder_lm(tokenizer, logits, lm=word_lm_scorer))
        refs.append(na_test['sentence'][i])
        # ------ save all LM predictions in a csv file ------ #
        df_lm = pd.DataFrame({'Reference': refs, 'Prediction': preds_lm})
        df_lm.to_csv(arguments.model_dir + 'results_decode_lm_' + arguments.lang + '.csv', index=False, sep='\t')


def decoding_lm_v2(arguments):
    model, processor, tokenizer, na_test, na_test_ref = pipeline(arguments.model_dir, arguments.test_tsv,
                                                                 arguments.lang)
    preds_lm = []
    refs = []
    for i in range(len(na_test)):
        input_dict = processor(na_test_ref["input_values"][i], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to("cuda")).logits
        preds_lm.append(
            beam_search_decoder_lm_v2(processor, tokenizer, logits, arguments.lm, arguments.alpha, arguments.beta))
        refs.append(na_test['sentence'][i])
        # ------ save all LM predictions in a csv file ------ #
        df_lm = pd.DataFrame({'Reference': refs, 'Prediction': preds_lm})
        df_lm.to_csv(arguments.model_dir + 'results_decode_lm_v2_' + arguments.lang + '.csv', index=False, sep='\t')


# ----------- Arguments ----------- #
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

beam_decode = subparsers.add_parser("decode",
                                    help="Generate predictions from fine-tuned model and print them.")
beam_decode.add_argument('--test_tsv', type=str, required=True,
                         help="Test .tsv file.")
beam_decode.add_argument('--model_dir', type=str, required=True,
                         help="Directory where the fine-tuned model is stored.")
beam_decode.add_argument('--json_path', type=str, required=True,
                         help="Directory to store json file.")
beam_decode.add_argument('--lang', type=str, required=True, choices={"japhug", "na"},
                         help="Language of the corpus.")
beam_decode.add_argument('--k_hyp', type=int, required=True,
                         help="K size of the top k hypotheses.")
beam_decode.set_defaults(func=decoding_top_k)

beam_decode_lm = subparsers.add_parser("decode_lm",
                                       help="Generate predictions from fine-tuned model and print them.")
beam_decode_lm.add_argument('--test_tsv', type=str, required=True,
                            help="Test .tsv file.")
beam_decode_lm.add_argument('--model_dir', type=str, required=True,
                            help="Directory where the fine-tuned model is stored.")
beam_decode_lm.add_argument('--logits_path', type=str, required=True,
                            help="Directory to the saved tensors.")
beam_decode_lm.add_argument('--lang', type=str, required=True, choices={"japhug", "na"},
                            help="Language of the corpus.")
beam_decode_lm.add_argument('--lm', type=str, required=True,
                            help="Word Ken language model.")
beam_decode_lm.add_argument('--alpha', type=int, required=True,
                            help="alpha lm parameter.")
beam_decode_lm.add_argument('--beta', type=int, required=True,
                            help="beta lm parameter.")
beam_decode_lm.set_defaults(func=decoding_lm)

beam_decode_lm_v2 = subparsers.add_parser("decode_lm_bis",
                                          help="Generate predictions from fine-tuned model and print them.")
beam_decode_lm_v2.add_argument('--test_tsv', type=str, required=True,
                               help="Test .tsv file.")
beam_decode_lm_v2.add_argument('--model_dir', type=str, required=True,
                               help="Directory where the fine-tuned model is stored.")
beam_decode_lm_v2.add_argument('--lang', type=str, required=True, choices={"japhug", "na"},
                               help="Language of the corpus.")
beam_decode_lm_v2.add_argument('--lm', type=str, required=True,
                               help="Word Ken language model.")
beam_decode_lm_v2.add_argument('--alpha', type=int, required=True,
                               help="alpha lm parameter.")
beam_decode_lm_v2.add_argument('--beta', type=int, required=True,
                               help="beta lm parameter.")
beam_decode_lm_v2.set_defaults(func=decoding_lm_v2)

arguments = parser.parse_args()
arguments.func(arguments)
