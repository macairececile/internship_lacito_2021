import pandas as pd
import argparse
from pathlib import Path
import evaluation as eval


def extract_pinyin_dataframe(dataframe, name_no_pinyin, name_pinyin):
    pinyin_ref = []
    pinyin_pred = []
    for index, row in dataframe.iterrows():
        a = ' '.join([word for word in row["Reference"].split() if word.startswith('@')])
        b = ' '.join([word for word in row["Prediction"].split() if word.startswith('@')])
        if len(a) > 0: pinyin_ref.append(a), pinyin_pred.append(b)
    dataframe["Reference_no_pinyin"] = dataframe["Reference"].apply(
        lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))
    dataframe["Prediction_no_pinyin"] = dataframe["Prediction"].apply(
        lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))
    df_pinyin = pd.DataFrame({'Reference': pinyin_ref, 'Prediction': pinyin_pred})
    df_pinyin.to_csv(name_pinyin + '.csv', index=False, sep='\t')
    df_nopinyin = pd.DataFrame(
        {'Reference': dataframe['Reference_no_pinyin'], 'Prediction': dataframe["Prediction_no_pinyin"]})
    df_nopinyin.to_csv(name_no_pinyin + '.csv', index=False, sep='\t')


def occurrence_frequency(ref, hyp):
    occurrences = {}
    wrong_prediction = {}
    for i, j in enumerate(ref):
        for k, l in enumerate(j):
            if l != '*' and l != 'S' and l != '@':
                if l not in occurrences.keys():
                    occurrences[l] = 1
                else:
                    occurrences[l] += 1
                if l not in wrong_prediction.keys():
                    wrong_prediction[l] = 0
                else:
                    if l != hyp[i][k]:
                        wrong_prediction[l] += 1
    occurrence_freq = {}
    for k, v in occurrences.items():
        if wrong_prediction[k] != 0:
            occurrence_freq[k] = wrong_prediction[k] / v
        else:
            occurrence_freq[k] = 0
    return occurrences, wrong_prediction, occurrence_freq


def pipeline_error_rate(csv_file, out_path):
    data = pd.read_csv(csv_file + '.csv', sep='\t')
    df, refs, preds = eval.compute_characters(data, out_path)
    occ, wrong, freq = occurrence_frequency(refs, preds)
    return df, refs, preds, occ, wrong, freq


def create_csv_occ_freq(init_data, no_pinyin_data, pinyin_data, out_path):
    df_init_data, init_data_refs, init_data_preds = eval.compute_characters(init_data, out_path)
    df_no_pinyin_data, no_pinyin_data_refs, no_pinyin_data_preds, occ_no_p, wrong_no_p, freq_no_p = pipeline_error_rate(
        no_pinyin_data, out_path)
    df_pinyin_data, pinyin_data_refs, pinyin_data_preds, occ_p, wrong_p, freq_p = pipeline_error_rate(pinyin_data,
                                                                                                      out_path)
    occ_i, wrong_i, freq_i = occurrence_frequency(init_data_refs, init_data_preds)
    all = [occ_i, wrong_i, freq_i, freq_p, freq_no_p]
    colnames = ['Occurrences', 'False_Predictions_all', 'Error_Rate_all', 'Error_Rate_pinyin', 'Error_Rate_no_pinyin']
    masterdf = pd.DataFrame()
    for i in all:
        df = pd.DataFrame([i]).T
        masterdf = pd.concat([masterdf, df], axis=1)

    # assign the column names
    masterdf.columns = colnames
    masterdf = masterdf.sort_values(by=['Error_Rate_pinyin'], ascending=False)
    # save to csv
    masterdf.to_csv(out_path + 'error_rate_pinyin_no_pinyin.csv', index=True, sep='\t')


def char_error_rate_occurrence_freq(args):
    out_path = Path('./analysis/')
    out_path.mkdir(exist_ok=True, parents=True)
    out_path = out_path.__str__() + '/'
    output_nopinyin = out_path + 'no_piyin_results.csv'
    output_pinyin = out_path + 'piyin_results.csv'
    data = eval.process_csv(args.input_file, 'japhug')
    extract_pinyin_dataframe(data, output_nopinyin, output_pinyin)
    create_csv_occ_freq(data, output_nopinyin, output_pinyin, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    error_rate = subparsers.add_parser("error_rate",
                                       help="")
    error_rate.add_argument('--input_file', type=str, required=True,
                            help="CSV result file with Reference and Prediction columns.")
    error_rate.set_defaults(func=char_error_rate_occurrence_freq)

    args = parser.parse_args()
    args.func(args)
