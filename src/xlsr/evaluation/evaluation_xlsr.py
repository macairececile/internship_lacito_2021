import statistics
from copy import copy
from functools import reduce
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import Levenshtein as lev
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import operator
import argparse
import difflib
from datasets import load_metric
import json
from pathlib import Path

wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

UNI_PHNS = {'q', 'p', 'ɭ', 'ɳ', 'h', 'ʐ', 'n', 'o', 'ɤ', 'ʝ', 'ɛ', 'g',
            'i', 'u', 'b', 'ɔ', 'ɯ', 'v', 'ɑ', 'l', 'ɖ', 'ɻ', 'ĩ', 'm',
            't', 'w', 'õ', 'ẽ', 'd', 'ɣ', 'ɕ', 'c', 'ʁ', 'ʑ', 'ʈ', 'ɲ', 'ɬ',
            's', 'ŋ', 'ə', 'e', 'æ', 'f', 'j', 'k', 'z', 'ʂ', 'S'}
BI_PHNS = {'dʑ', 'ẽ', 'ɖʐ', 'w̃', 'æ̃', 'qʰ', 'i͂', 'tɕ', 'v̩', 'o̥', 'ts', 'sʰ', 'ɕʰ',
           'ɻ̩', 'ã', 'ə̃', 'ṽ', 'pʰ', 'tʰ', 'ɤ̃', 'ʈʰ', 'ʈʂ', 'ɑ̃', 'ɻ̃', 'kʰ', 'hʰ', 'ʂʰ',
           'ĩ', 'õ', 'dz', "ɻ̍", "wæ", "wɑ", "wɤ", "jæ", "jɤ", "jo", "ʋ̩", 'ə…', 'pʰ', '*̩'}
FILLERS = {"əəə…", "mmm…"}
TRI_PHNS = {"tɕʰ", "ʈʂʰ", "tsʰ", "ṽ̩", "ṽ̩", "ɻ̩̃", "wæ̃", "w̃æ", "ʋ̩̃", "ɻ̩̃", 'tʂʰ', 'ʈʰ̩', 'tʰ̩'}
UNI_TONES = {"˩", "˥", "˧"}
BI_TONES = {"˧˥", "˩˥", "˩˧", "˧˩"}
TONES = UNI_TONES.union(BI_TONES)


def compute_levenshtein_distance(dataframe, tones=True):
    """Evaluation of edition distance between words per sentence by using the levenshtein distance."""
    dist = []
    lev_all = []
    num_words = []
    if tones:
        dataframe['Ref_words'] = dataframe['Reference'].apply(lambda x: x.split(' '))
        dataframe['Pred_words'] = dataframe['Prediction'].apply(lambda x: x.split(' '))
        for index, row in dataframe.iterrows():
            num_words.append(len(row["Pred_words"]) - len(row["Ref_words"]))
            lev_all.append(lev.ratio(row["Reference"], row["Prediction"]))
            lev_d = []
            for i, j in enumerate(row["Ref_words"]):
                if i < len(row["Pred_words"]):
                    lev_d.append(lev.ratio(row["Ref_words"][i], row["Pred_words"][i]))
                else:
                    lev_d.append(lev.ratio(row["Ref_words"][i], ''))
            dist.append(lev_d)
        dataframe['Diff_num_words'] = num_words
        dataframe['Lev_distance'] = lev_all
        print('WER_huggingface : ',
              wer_metric.compute(predictions=dataframe["Reference"], references=dataframe["Prediction"]))
        print('CER_huggingface : ',
              cer_metric.compute(predictions=dataframe["Reference"], references=dataframe["Prediction"]))
        print('WER_lev : ', 1 - st.mean([j for a in dist for j in a]))
        print('CER_lev : ', 1 - st.mean(lev_all))
        dataframe['Lev_distance_words'] = dist
        dataframe['Average_lev_dist_words'] = dataframe['Lev_distance_words'].apply(lambda x: st.mean(x))
    else:
        dataframe['Ref_words_noTones'] = dataframe['Ref_noTones'].apply(lambda x: x.split(' '))
        dataframe['Pred_words_noTones'] = dataframe['Pred_noTones'].apply(lambda x: x.split(' '))
        for index, row in dataframe.iterrows():
            num_words.append(len(row["Pred_words_noTones"]) - len(row["Ref_words_noTones"]))
            lev_all.append(lev.ratio(row["Ref_noTones"], row["Pred_noTones"]))
            lev_d = []
            for i, j in enumerate(row["Ref_words_noTones"]):
                if i < len(row["Pred_words_noTones"]):
                    lev_d.append(lev.ratio(row["Ref_words_noTones"][i], row["Pred_words_noTones"][i]))
                else:
                    lev_d.append(lev.ratio(row["Ref_words_noTones"][i], ''))
            dist.append(lev_d)
        dataframe['Diff_num_words'] = num_words
        dataframe['Lev_distance_noTones'] = lev_all
        dataframe['Lev_distance_words_notones'] = dist
        print('WER_huggingface : ',
              wer_metric.compute(predictions=dataframe["Ref_noTones"], references=dataframe["Pred_noTones"]))
        print('CER_huggingface : ',
              cer_metric.compute(predictions=dataframe["Ref_noTones"], references=dataframe["Pred_noTones"]))
        print('WER_lev : ', 1 - st.mean([j for a in dist for j in a]))
        print('CER_lev : ', 1 - st.mean(lev_all))
        dataframe['Average_lev_dist_notones'] = dataframe['Lev_distance_words_notones'].apply(lambda x: st.mean(x))
    return dataframe


def compute_without_tones(dataframe):
    """Evaluation without considering the tones"""
    no_tones = []
    no_tones_2 = []
    for index, row in dataframe.iterrows():
        no_tones.append(reduce(lambda a, b: a.replace(b, ''), TONES, row["Reference"]))
        no_tones_2.append(reduce(lambda a, b: a.replace(b, ''), TONES, row["Prediction"]))
    dataframe['Ref_noTones'] = no_tones
    dataframe['Pred_noTones'] = no_tones_2
    return compute_levenshtein_distance(dataframe, tones=False)


def compute_characters(dataframe, out_path):
    f_score = []
    ref_g = []
    hyp_g = []
    precision = []
    recall = []
    f = open(out_path + "latex.txt", "w")
    S_insert = 0
    S_delete = 0
    for index, row in dataframe.iterrows():
        edit = lev.editops(row['Prediction'], row['Reference'])
        ind = 0
        ind_bis = 0
        hyp = [i for i in row['Prediction']]
        ref = [i for i in row['Reference']]
        hyp_out = copy(hyp)
        ref_out = copy(ref)
        for i, j in enumerate(edit):
            if edit[i][0] == 'insert':
                if ref_out[edit[i][2]] == ' ': S_insert += 1
                hyp.insert(edit[i][1] + ind_bis, '*')
                ind_bis += 1
                ref_out[edit[i][2]] = '\\hl{' + ref_out[edit[i][2]] + '}'
            elif edit[i][0] == 'delete':
                if hyp_out[edit[i][1]] == ' ': S_delete += 1
                hyp_out[edit[i][1]] = '\\hl{' + hyp_out[edit[i][1]] + '}'
                if len(ref) > edit[i][1] + ind:
                    ref.insert(edit[i][2] + ind, '*')
                    ind += 1
                else:
                    ref.append('*')
            elif edit[i][0] == 'replace':
                hyp_out[edit[i][1]] = '\\hl{' + hyp_out[edit[i][1]] + '}'
                ref_out[edit[i][2]] = '\\hl{' + ref_out[edit[i][2]] + '}'
        ref_out = ''.join(ref_out).replace('\\hl{ }', ' ')
        hyp_out = ''.join(hyp_out).replace('\\hl{ }', ' ')
        f.write('Ref: ' + ref_out + '\\n' + 'Hyp: ' + hyp_out + '\\n\\midrule \n')
        ref = ['S' if x == ' ' else x for x in ref]
        hyp = ['S' if x == ' ' else x for x in hyp]
        ref_g.append(ref)
        hyp_g.append(hyp)
        f_score.append(round(f1_score(ref, hyp, average="macro"), 3))
        precision.append(round(precision_score(ref, hyp, average="macro", zero_division=1), 3))
        recall.append(round(recall_score(ref, hyp, average="macro", zero_division=1), 3))
    # print('S_insert: ', S_insert)
    # print('S_delete: ', S_delete)
    f.close()
    dataframe['Ref_char'] = ref_g
    dataframe['Pred_char'] = hyp_g
    dataframe['F_score_char'] = f_score
    dataframe['Precision_char'] = precision
    dataframe['Recall_char'] = recall
    return dataframe, ref_g, hyp_g


def preprocessing_text(sentence_r, sentence_h):
    if sentence_r[:5] == 'əĩə…' or sentence_h[:5] == 'əĩə…':
        return sentence_r[:5], sentence_h[:5], sentence_r[5:], sentence_h[5:]
    if sentence_r[:4] in FILLERS or sentence_h[:4] in FILLERS:
        return sentence_r[:4], sentence_h[:4], sentence_r[4:], sentence_h[4:]
    if sentence_r[:3] in TRI_PHNS or sentence_h[:3] in TRI_PHNS:
        return sentence_r[:3], sentence_h[:3], sentence_r[3:], sentence_h[3:]
    if sentence_r[:2] in BI_PHNS or sentence_h[:2] in BI_PHNS:
        return sentence_r[:2], sentence_h[:2], sentence_r[2:], sentence_h[2:]
    if sentence_r[:2] in BI_TONES or sentence_h[:2] in BI_TONES:
        return sentence_r[:2], sentence_h[:2], sentence_r[2:], sentence_h[2:]
    if sentence_r[0] in UNI_PHNS or sentence_h[0] in UNI_PHNS:
        return sentence_r[0], sentence_h[0], sentence_r[1:], sentence_h[1:]
    if sentence_r[0] in UNI_TONES or sentence_h[0] in UNI_TONES:
        return sentence_r[0], sentence_h[0], sentence_r[1:], sentence_h[1:]
    if sentence_r[0] == '*' or sentence_h[0] == '*':
        return sentence_r[0], sentence_h[0], sentence_r[1:], sentence_h[1:]


def filter_for_phonemes(sentence_r, sentence_h):
    """ Returns a sequence of phonemes """
    phonemes_R = []
    phonemes_H = []
    while sentence_r != '':
        phoneme_r, phoneme_h, sentence_r, sentence_h = preprocessing_text(sentence_r, sentence_h)
        phonemes_R.append(phoneme_r)
        phonemes_H.append(phoneme_h)
    return phonemes_R, phonemes_H


def compute_phonemes(dataframe, ref, pred):
    ref_all = []
    hyp_all = []
    ratio = []
    for i in range(len(ref)):
        p_ref, p_hyp = filter_for_phonemes(''.join(ref[i]), ''.join(pred[i]))
        str_p_ref = ' '.join(p_ref)
        str_p_hyp = ' '.join(p_hyp)
        s = difflib.SequenceMatcher(None, str_p_ref, str_p_hyp)
        ratio.append(s.ratio())
        ref_all.append(p_ref)
        hyp_all.append(p_hyp)
    dataframe['Ref_phon'] = ref_all
    dataframe['Pred_phon'] = hyp_all
    dataframe['Phonemes_lev_dist'] = ratio
    print('PER : ', 1 - statistics.mean(ratio))
    return dataframe, ref_all, hyp_all


def confusion_matrix_phoneme(ref, pred, phoneme, out_path):
    ref_all = [x for y in ref for x in y]
    hyp_all = [x for y in pred for x in y]
    select_ref = []
    select_hyp = []
    for i, j in enumerate(ref_all):
        if j == phoneme:
            select_ref.append(ref_all[i])
            select_hyp.append(hyp_all[i])
    plt.rcParams.update({'font.size': 25, 'xtick.labelsize': 'small', 'ytick.labelsize': 'small'})
    labels = list(set(select_ref) | set(select_hyp))
    cm = confusion_matrix(select_ref, select_hyp, labels=labels)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(cm, annot=True, ax=ax, cmap="OrRd", fmt="d")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticklabels(labels, rotation=45, ha='right', minor=False)
    ax.set_yticklabels(labels, rotation=0)
    plt.xlabel('Predicted labels', labelpad=20)
    plt.ylabel('Referenced labels', labelpad=20)
    plt.tight_layout()
    plt.show()
    conf_matrix = pd.crosstab(np.array(ref_all), np.array(hyp_all), rownames=['Reference'], colnames=['Hypothesis'],
                              margins=True)
    results = {}
    d = conf_matrix.to_dict(orient='records')
    ind = list(conf_matrix.index.values)
    for i, j in enumerate(ind):
        if j != 'All':
            del d[i]['All']
            if j in d[i].keys():
                del d[i][j]
            results[j] = d[i]
    best_wrong_associations = {}
    for k, v in results.items():
        best_wrong_associations[k] = max(v.items(), key=operator.itemgetter(1))[0]
    with open(out_path + 'best_wrong_predictions.txt', 'w') as file:
        file.write(json.dumps(best_wrong_associations))


def process_csv(path, lang):
    df = pd.read_csv(path, sep='\t')
    if lang == 'na':
        df['Reference'] = df['Reference'].apply(lambda x: x.replace('|', ' '))
        df['Prediction'] = df['Prediction'].apply(lambda x: x.replace('[UNK]', '*'))
        df['Prediction'] = df['Prediction'].apply(lambda x: x.replace('|', ' '))
    return df


def eval_lev(args):
    out_path = Path('./analysis/')
    out_path.mkdir(exist_ok=True, parents=True)
    out_path = out_path.__str__() + '/'
    data = process_csv(args.input_file, args.lang)
    results = compute_levenshtein_distance(data)
    results.to_csv(out_path + 'results_analysis_lev_dist.csv', sep='\t', index=False)


def eval_lev_notones(args):
    out_path = Path('./analysis/')
    out_path.mkdir(exist_ok=True, parents=True)
    out_path = out_path.__str__() + '/'
    data = process_csv(args.input_file, args.lang)
    results = compute_without_tones(data)
    results.to_csv(out_path + 'results_analysis_lev_dist_no_tones.csv', sep='\t', index=False)


def eval_char(args):
    out_path = Path('./analysis/')
    out_path.mkdir(exist_ok=True, parents=True)
    out_path = out_path.__str__() + '/'
    data = process_csv(args.input_file, args.lang)
    data2, refs, preds = compute_characters(data, out_path)
    if args.lang == 'na':
        results, refs, preds = compute_phonemes(data2, refs, preds)
        results.to_csv(out_path + 'results_analysis_char.csv', sep='\t', index=False)
        if args.confusion_matrix:
            confusion_matrix_phoneme(refs, preds, args.phone, out_path)
    elif args.lang == 'japhug':
        data2.to_csv(out_path + 'results_analysis_char.csv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    lev_dist = subparsers.add_parser("lev_dist",
                                     help="Compute the levenshtein distance between each reference and its corresponding prediction (all sentence and sentence splitted by words")
    lev_dist.add_argument('--input_file', type=str, required=True,
                          help="CSV result file with Reference and Prediction columns.")
    lev_dist.add_argument('--lang', type=str, required=True, choices=['japhug', 'na'],
                          help="Language of the result file.")
    lev_dist.set_defaults(func=eval_lev)

    lev_dist_notones = subparsers.add_parser("lev_dist_notones",
                                             help="Compute the levenshtein distance between each reference and its corresponding prediction without tones (all sentence and sentence splitted by words")
    lev_dist_notones.add_argument('--input_file', type=str, required=True,
                                  help="CSV result file with Reference and Prediction columns.")
    lev_dist_notones.add_argument('--lang', type=str, required=True, choices=['na'],
                                  help="Language of the result file.")
    lev_dist_notones.set_defaults(func=eval_lev_notones)

    eval_phon = subparsers.add_parser("eval_char",
                                      help="Analysis of character similarities between references and predictions.")
    eval_phon.add_argument('--input_file', type=str, required=True,
                           help="CSV result file with Reference and Prediction columns.")
    eval_phon.add_argument('--lang', type=str, required=True, choices=['japhug', 'na'],
                           help="Language of the result file.")
    eval_phon.add_argument('--confusion_matrix', type=bool, required=False,
                           help="Show the confusion matrix for phonemes (True or False).")
    eval_phon.add_argument('--phone', type=str, required=False,
                           help="Phoneme to print the confusion matrix.")
    eval_phon.set_defaults(func=eval_char)

    args = parser.parse_args()
    args.func(args)
