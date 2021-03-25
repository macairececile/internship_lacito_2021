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

MISC_SYMBOLS = [' ̩', '~', '=', ':', 'F', '¨', '↑', '“', '”', '…', '«', '»',
                'D', 'a', 'ː', '#', '$', "‡", "˞"]
BAD_NA_SYMBOLS = ['D', 'F', '~', '…', '=', '↑', ':']
PUNC_SYMBOLS = [',', '!', '.', ';', '?', "'", '"', ':', '«', '»', '“', '”', "ʔ", "+"]
UNI_PHNS = {'q', 'p', 'ɭ', 'ɳ', 'h', 'ʐ', 'n', 'o', 'ɤ', 'ʝ', 'ɛ', 'g',
            'i', 'u', 'b', 'ɔ', 'ɯ', 'v', 'ɑ', 'l', 'ɖ', 'ɻ', 'ĩ', 'm',
            't', 'w', 'õ', 'ẽ', 'd', 'ɣ', 'ɕ', 'c', 'ʁ', 'ʑ', 'ʈ', 'ɲ', 'ɬ',
            's', 'ŋ', 'ə', 'e', 'æ', 'f', 'j', 'k', 'z', 'ʂ'}
BI_PHNS = {'dʑ', 'ẽ', 'ɖʐ', 'w̃', 'æ̃', 'qʰ', 'i͂', 'tɕ', 'v̩', 'o̥', 'ts',
           'ɻ̩', 'ã', 'ə̃', 'ṽ', 'pʰ', 'tʰ', 'ɤ̃', 'ʈʰ', 'ʈʂ', 'ɑ̃', 'ɻ̃', 'kʰ',
           'ĩ', 'õ', 'dz', "ɻ̍", "wæ", "wɑ", "wɤ", "jæ", "jɤ", "jo", "ʋ̩"}
FILLERS = {"əəə…", "mmm…"}
TRI_PHNS = {"tɕʰ", "ʈʂʰ", "tsʰ", "ṽ̩", "ṽ̩", "ɻ̩̃", "wæ̃", "w̃æ", "ʋ̩̃", "ɻ̩̃"}
UNI_TONES = {"˩", "˥", "˧"}
BI_TONES = {"˧˥", "˩˥", "˩˧", "˧˩"}
TONES = UNI_TONES.union(BI_TONES)


def preprocessing_text(sentence):
    if sentence[:4] == '<SP>':
        return sentence[:4], sentence[4:]
    if sentence[:4] in ["əəə…", "mmm…"]:
        return sentence[:4], sentence[4:]
    if sentence.startswith("ə…"):
        return "əəə…", sentence[2:]
    if sentence.startswith("m…"):
        return "mmm…", sentence[2:]
    if sentence.startswith("mm…"):
        return "mmm…", sentence[3:]
    if sentence[:3] == "wæ̃":
        return "w̃æ", sentence[3:]
    if sentence[:3] == "ṽ̩":
        return "ṽ̩", sentence[3:]
    if sentence[:3] in TRI_PHNS:
        return sentence[:3], sentence[3:]
    if sentence[:2] in BI_PHNS:
        return sentence[:2], sentence[2:]
    if sentence[:2] == "˧̩":
        return "˧", sentence[2:]
    if sentence[:2] == "˧̍":
        return "˧", sentence[2:]
    if sentence[0] in UNI_PHNS:
        return sentence[0], sentence[1:]
    if sentence[:2] in BI_TONES:
        return sentence[:2], sentence[2:]
    if sentence[0] in UNI_TONES:
        return sentence[0], sentence[1:]
    if sentence[0] in MISC_SYMBOLS:
        # We assume these symbols cannot be captured.
        return None, sentence[1:]
    if sentence[0] in BAD_NA_SYMBOLS:
        return None, sentence[1:]
    if sentence[0] in PUNC_SYMBOLS:
        return None, sentence[1:]
    if sentence[0] in ["-", "ʰ", "/"]:
        return None, sentence[1:]
    if sentence[0] in {"<", ">"}:
        # We keep everything literal, thus including what is in <>
        # brackets; so we just remove these tokens"
        return None, sentence[1:]
    if sentence[0] == "[":
        if sentence.find("]") == len(sentence) - 1:
            return None, ""
        else:
            return None, sentence[sentence.find("]") + 1:]
    if sentence[0] in {" ", "\t", "\n"}:
        # Return a space char so that it can be identified in word segmentation
        # processing.
        return " ", sentence[1:]
    if sentence[0] == "|" or sentence[0] == "ǀ" or sentence[0] == "◊":
        return "|", sentence[1:]
    if sentence[0] == '*':
        return "*", sentence[1:]
    if sentence[0] in "()":
        return None, sentence[1:]


def filter_for_phonemes(sentence):
    """ Returns a sequence of phonemes and pipes (word delimiters). Tones,
        syllable boundaries, whitespace are all removed."""
    filtered_sentence = []
    phonemes = []
    while sentence != "":
        phoneme, sentence = preprocessing_text(sentence)
        phonemes.append(phoneme)
        if phoneme != " ":
            filtered_sentence.append(phoneme)
    filtered_sentence = [item for item in filtered_sentence if item != None]
    return " ".join(filtered_sentence), phonemes


# IF WANT TO CUT BY PHONEMES
def final_text_phonemes(batch):
    if "BEGAIEMENT" in batch['sentence']:
        batch['sentence'] = batch['sentence'].replace('BEGAIEMENT', "")
    s, p = filter_for_phonemes(batch['sentence'])
    s = s.replace(' ', '|')
    s = s.replace('||', '|')
    s = s.replace('|||', '|')
    batch['sentence'] = s
    p = list(filter(None, p))
    batch['phonemes'] = p
    return batch


# IF WANT TO KEEP WORDS
def final_text_words(batch):
    if "BEGAIEMENT" in batch['sentence']:
        batch['sentence'] = batch['sentence'].replace('BEGAIEMENT', "")
    s, p = filter_for_phonemes(batch['sentence'])
    batch['sentence'] = s.replace(' ', '')
    p = list(filter(None, p))
    batch['phonemes'] = p
    return batch


def extract_phonemes(data):
    if "BEGAIEMENT" in data:
        data = data.replace('BEGAIEMENT', "")
    s, p = filter_for_phonemes(data)
    if '|' in p: p.remove('|')
    p = list(filter(None, p))
    return p


def evaluation_levenshtein_distance(dataframe, tones=True):
    """Evaluation of edition distance between words per sentence by using the levenshtein distance."""
    dist = []
    diff_phon = []
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


def evaluation_without_tones(dataframe):
    """Evaluation without considering the tones"""
    no_tones = []
    no_tones_2 = []
    for index, row in dataframe.iterrows():
        no_tones.append(reduce(lambda a, b: a.replace(b, ''), TONES, row["Reference"]))
        no_tones_2.append(reduce(lambda a, b: a.replace(b, ''), TONES, row["Prediction"]))
    dataframe['Ref_noTones'] = no_tones
    dataframe['Pred_noTones'] = no_tones_2
    return evaluation_levenshtein_distance(dataframe, tones=False)


def evaluation_phonemes(dataframe):
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
        if len(ref) == len(hyp):
            all_ref.extend(ref)
            all_pred.extend(hyp)
            f_score.append(round(f1_score(ref, hyp, average="macro"), 3))
            precision.append(round(precision_score(ref, hyp, average="macro", zero_division=1), 3))
            recall.append(round(recall_score(ref, hyp, average="macro", zero_division=1), 3))
        else:
            f_score.append(None)
            precision.append(None)
            recall.append(None)
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


def global_evaluation(dataframe):
    dataframe['Reference'] = dataframe['Reference'].apply(lambda x: x[:-1].replace('|', ' '))
    dataframe['Prediction'] = dataframe['Prediction'].apply(lambda x: x.replace('[UNK]', '*'))
    evaluation_levenshtein_distance(df)
    evaluation_without_tones(df)
    dataframe, ref, pred = evaluation_phonemes(df)
    return dataframe, ref, pred


def confusion_matrix_all_phonemes(ref, pred):
    results = {}
    confusion_matrix = pd.crosstab(np.array(ref), np.array(pred),
                                   rownames=['Reference'], colnames=['Hypothesis'], margins=True)
    d = confusion_matrix.to_dict(orient='records')
    ind = list(confusion_matrix.index.values)
    for i, j in enumerate(ind):
        if j != 'All':
            del d[i]['All']
            if j in d[i].keys():
                del d[i][j]
            results[j] = d[i]
    best_wrong_associations = {}
    for k, v in results.items():
        best_wrong_associations[k] = max(v.items(), key=operator.itemgetter(1))[0]

    print(best_wrong_associations)
    # sns.heatmap(confusion_matrix, annot=True, yticklabels=list(set(ref)), xticklabels=list(set(pred)))
    # plt.show()

def stats(dataframe):
    print('Lev distance mean: ', dataframe["Lev_distance"].mean())
    print('Lev distance no tones mean: ', dataframe["Lev_distance_noTones"].mean())
    print('Average_lev_dist_words mean: ', dataframe["Average_lev_dist_words"].mean())
    print('Average_lev_dist_notones mean: ', dataframe["Average_lev_dist_notones"].mean())


if __name__ == '__main__':
    df = pd.read_csv('/home/cmacaire/Desktop/training/test_4/results.csv')
    new_df, ref, pred = global_evaluation(df)
    confusion_matrix_all_phonemes(ref, pred)
    print(new_df.loc[0])
    print(new_df.loc[1])
    print(new_df.loc[4])
    stats(new_df)
