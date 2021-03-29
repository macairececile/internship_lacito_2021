from functools import reduce
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay
import Levenshtein as lev
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import operator
import argparse

UNI_TONES = {"˩", "˥", "˧"}
BI_TONES = {"˧˥", "˩˥", "˩˧", "˧˩"}
TONES = UNI_TONES.union(BI_TONES)


def compute_levenshtein_distance(dataframe, tones=True):
    """Evaluation of edition distance between words per sentence by using the levenshtein distance."""
    dist = []
    if tones:
        dataframe['Ref_words'] = dataframe['Reference'].apply(lambda x: x.split(' '))
        dataframe['Pred_words'] = dataframe['Prediction'].apply(lambda x: x.split(' '))
        for index, row in dataframe.iterrows():
            dataframe['Lev_distance'] = lev.ratio(' '.join(row["Ref_words"]), ' '.join(row["Pred_words"]))
            lev_d = []
            for i, j in enumerate(row["Ref_words"]):
                if i < len(row["Pred_words"]):
                    lev_d.append(lev.ratio(row["Ref_words"][i], row["Pred_words"][i]))
                else:
                    lev_d.append(lev.ratio(row["Ref_words"][i], ''))
            dist.append(lev_d)
        # dataframe['Diff_phon'] = diff_phon
        dataframe['Lev_distance_words'] = dist
        dataframe['Average_lev_dist_words'] = dataframe['Lev_distance_words'].apply(lambda x: st.mean(x))
    else:
        dataframe['Ref_words_noTones'] = dataframe['Ref_noTones'].apply(lambda x: x.split(' '))
        dataframe['Pred_words_noTones'] = dataframe['Pred_noTones'].apply(lambda x: x.split(' '))
        for index, row in dataframe.iterrows():
            dataframe['Lev_distance_noTones'] = lev.ratio(' '.join(row["Ref_noTones"]), ' '.join(row["Pred_noTones"]))
            lev_d = []
            for i, j in enumerate(row["Ref_words_noTones"]):
                if i < len(row["Pred_words_noTones"]):
                    lev_d.append(lev.ratio(row["Ref_words_noTones"][i], row["Pred_words_noTones"][i]))
                else:
                    lev_d.append(lev.ratio(row["Ref_words_noTones"][i], ''))
            dist.append(lev_d)
        dataframe['Lev_distance_words_notones'] = dist
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


def compute_phonemes(dataframe):
    f_score = []
    ref_g = []
    hyp_g = []
    precision = []
    recall = []
    all_ref = []
    all_pred = []
    for index, row in dataframe.iterrows():
        edit = lev.editops(row['Prediction'], row['Reference'])
        # print(edit)
        ind = 0
        hyp = [i for i in row['Prediction']]
        ref = [i for i in row['Reference']]
        if len(hyp) < len(ref):
            for i, j in enumerate(edit):
                if edit[i][0] == 'insert':
                    ind += 1
                    hyp.insert(edit[i][1], '*')
                elif edit[i][0] == 'delete':
                    if len(ref) > edit[i][2] + ind:
                        ref.insert(edit[i][2] + ind, '*')
                        ind += 1
                    else:
                        ref.append('*')
        else:
            for i, j in enumerate(edit):
                if edit[i][0] == 'insert':
                    ind += 1
                    hyp.insert(edit[i][1], '*')
                if edit[i][0] == 'delete':
                    if len(ref) > edit[i][2] + ind:
                        ref.insert(edit[i][2] + ind, '*')
                        ind += 1
                    else:
                        ref.append('*')
        ref = ['<SP>' if x == ' ' else x for x in ref]
        hyp = ['<SP>' if x == ' ' else x for x in hyp]
        ref_g.append(ref)
        hyp_g.append(hyp)
        all_ref.extend(ref)
        all_pred.extend(hyp)
        f_score.append(round(f1_score(ref, hyp, average="macro"), 3))
        precision.append(round(precision_score(ref, hyp, average="macro", zero_division=1), 3))
        recall.append(round(recall_score(ref, hyp, average="macro", zero_division=1), 3))
        # confusion_matrix = pd.crosstab(np.array(ref), np.array(hyp),
        #                                rownames=['Reference'], colnames=['Hypothesis'])
        # sns.heatmap(confusion_matrix, annot=True)
        # plt.show()
    dataframe['Ref_phonemes'] = ref_g
    dataframe['Pred_phonemes'] = hyp_g
    dataframe['F_score_phonemes'] = f_score
    dataframe['Precision_phonemes'] = precision
    dataframe['Recall_phonemes'] = recall
    return dataframe, all_ref, all_pred


def confusion_matrix_all_phonemes(ref, pred):
    confusion_matrix = pd.crosstab(np.array(ref), np.array(pred),
                                   rownames=['Reference'], colnames=['Hypothesis'], margins=True)
    sns.heatmap(confusion_matrix, annot=True)
    plt.show()
    # results = {}
    # d = confusion_matrix.to_dict(orient='records')
    # ind = list(confusion_matrix.index.values)
    # for i, j in enumerate(ind):
    #     if j != 'All':
    #         del d[i]['All']
    #         if j in d[i].keys():
    #             del d[i][j]
    #         results[j] = d[i]
    # best_wrong_associations = {}
    # for k, v in results.items():
    #     best_wrong_associations[k] = max(v.items(), key=operator.itemgetter(1))[0]
    # print(best_wrong_associations)


def process_csv(path):
    df = pd.read_csv(path, sep='\t')
    df['Reference'] = df['Reference'].apply(lambda x: x[:-1].replace('|', ' '))
    df['Prediction'] = df['Prediction'].apply(lambda x: x.replace('[UNK]', '*'))
    return df


def eval_lev(args):
    data = process_csv(args.input_file)
    results = compute_levenshtein_distance(data)
    results.to_csv('results_analysis_lev_dist.csv', sep='\t', index=False)


def eval_lev_notones(args):
    data = process_csv(args.input_file)
    results = compute_without_tones(data)
    results.to_csv('results_analysis_lev_dist_no_tones.csv', sep='\t', index=False)


def eval_phonemes(args):
    data = process_csv(args.input_file)
    results, refs, preds = compute_phonemes(data)
    results.to_csv('results_analysis_phonemes.csv', sep='\t', index=False)
    if args.confusion_matrix:
        confusion_matrix_all_phonemes(refs, preds)


def stats(args):
    data = process_csv(args.input_file)
    print('Lev distance mean: ', data["Lev_distance"].mean())
    print('Lev distance no tones mean: ', data["Lev_distance_noTones"].mean())
    print('Average_lev_dist_words mean: ', data["Average_lev_dist_words"].mean())
    print('Average_lev_dist_notones mean: ', data["Average_lev_dist_notones"].mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    lev_dist = subparsers.add_parser("lev_dist",
                                     help="Compute the levenshtein distance between each reference and its corresponding prediction (all sentence and sentence splitted by words")
    lev_dist.add_argument('--input_file', type=str, required=True,
                          help="CSV result file with Reference and Prediction columns.")
    lev_dist.set_defaults(func=eval_lev)

    lev_dist_notones = subparsers.add_parser("lev_dist_notones",
                                             help="Compute the levenshtein distance between each reference and its corresponding prediction without tones (all sentence and sentence splitted by words")
    lev_dist_notones.add_argument('--input_file', type=str, required=True,
                                  help="CSV result file with Reference and Prediction columns.")
    lev_dist_notones.set_defaults(func=eval_lev_notones)

    eval_phon = subparsers.add_parser("eval_phonemes",
                                      help="Analysis of phoneme similarities between references and predictions.")
    eval_phon.add_argument('--input_file', type=str, required=True,
                           help="CSV result file with Reference and Prediction columns.")
    eval_phon.add_argument('--confusion_matrix', type=bool, required=True,
                           help="Show the confusion matrix for phonemes (True or False).")
    eval_phon.set_defaults(func=eval_phonemes)

    statistics = subparsers.add_parser("stats",
                                       help="Show some statistics computed from previous results.")
    statistics.add_argument('--input_file', type=str, required=True,
                            help="CSV of the analyzed result file.")
    statistics.set_defaults(func=stats)

    args = parser.parse_args()
    args.func(args)
