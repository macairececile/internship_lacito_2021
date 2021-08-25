import argparse
import operator
import os
import librosa
import pandas as pd
from random import sample


def get_parser():
    parser = argparse.ArgumentParser(description="Select specific examples")

    parser.add_argument('--path_files', type=str, required=True,
                        help="")
    parser.add_argument('--n', type=int, required=True,
                        help="")
    parser.add_argument('--path_audio', type=str, required=True,
                        help="")

    return parser


def extract_info(src_file):
    with open(src_file, 'r') as f:
        lines = f.readlines()
    return [s.split(' ') for s in lines if s]


def number_unique_clusters(phones):
    unique_clust = {}
    for s, a in enumerate(phones):
        p = len(set(a))
        unique_clust[s] = p
    od = dict(sorted(unique_clust.items(), key=operator.itemgetter(1), reverse=True))
    return od


def random_selections(phones):
    num_sentences = len(phones)
    random_numbers = sample(range(0, num_sentences), num_sentences)
    random = {}
    for i in random_numbers:
        random[i] = phones[i]
    return random


def select_n_examples(path, unique_c, n, path_audio, split):
    output_path = path + 'selections_random_' + str(n) + '/'
    os.makedirs(output_path, exist_ok=True)
    ltr_file = open(path + split + '.ltr', 'r').readlines()
    phn_file = open(path + split + '.phn', 'r').readlines()
    wrd_file = open(path + split + '.wrd', 'r').readlines()
    new_ltr = open(output_path + split + '.ltr', 'w')
    new_phn = open(output_path + split + '.phn', 'w')
    new_wrd = open(output_path + split + '.wrd', 'w')
    time = 0
    audio_files = pd.read_csv(path + split + '.tsv', sep='\t')
    data_tsv = pd.DataFrame(columns=audio_files.columns)
    for k, v in unique_c.items():
        if time <= n:
            time += round(librosa.get_duration(filename=path_audio + audio_files.iat[k, 0]), 2) / 60
            new_ltr.write(ltr_file[k])
            new_phn.write(phn_file[k])
            new_wrd.write(wrd_file[k])
            # print(audio_files.iloc[k])
            data_tsv.loc[len(data_tsv)] = [audio_files.iat[k, 0], audio_files.iat[k, 1]]
        else:
            break
    print(data_tsv)
    data_tsv.to_csv(output_path + split + '.tsv', sep='\t', index=False)
    new_ltr.close()
    new_wrd.close()
    new_phn.close()


def main():
    parser = get_parser()
    args = parser.parse_args()
    split = ['train', 'test', 'valid']
    # split = ['test']
    for i in split:
        phones = extract_info(args.path_files + i + '.src')
        # unique = number_unique_clusters(phones)
        random_select = random_selections(phones)
        select_n_examples(args.path_files, random_select, args.n, args.path_audio, i)


if __name__ == '__main__':
    main()
