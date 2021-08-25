from datasets import load_dataset
import preprocessing_text_na as prep
import preprocessing_text_japhug_2 as prep2
import re
import pandas as pd


def load_and_process(train_tsv, val_tsv, test_tsv):
    train_data = load_dataset('csv', data_files=[train_tsv], delimiter='\t')
    val_data = load_dataset('csv', data_files=[val_tsv], delimiter='\t')
    test_data = load_dataset('csv', data_files=[test_tsv], delimiter='\t')
    train_data = train_data['train']
    val_data = val_data['train']
    test_data = test_data['train']
    train_data = train_data.map(prep2.final_text_words)
    val_data = val_data.map(prep2.final_text_words)
    test_data = test_data.map(prep2.final_text_words)
    return train_data, val_data, test_data


def create_word_files(train, val, test):
    f_train = open('train.wrd', 'w')
    f_val = open('valid.wrd', 'w')
    f_test = open('test.wrd', 'w')
    for el in train['sentence']:
        f_train.write(el + '\n')
    for i in val['sentence']:
        f_val.write(i + '\n')
    for j in test['sentence']:
        f_test.write(j + '\n')


def create_letter_files(train, val, test):
    f_train = open('train.ltr', 'w')
    f_val = open('valid.ltr', 'w')
    f_test = open('test.ltr', 'w')
    for el in train['sentence']:
        f_train.write(" ".join(list(el.strip().replace(" ", "|"))) + " |" + '\n')
    for i in val['sentence']:
        f_val.write(" ".join(list(i.strip().replace(" ", "|"))) + " |" + '\n')
    for j in test['sentence']:
        f_test.write(" ".join(list(j.strip().replace(" ", "|"))) + " |" + '\n')


def clear_phones(sentence):
    sentence[:] = [x if x != '|' else ' ' for x in sentence]
    sentence = " ".join(sentence)
    sentence = re.sub(' +', ' ', sentence)
    return sentence


def create_phones_files(train, val, test):
    f_train = open('train.phn', 'w')
    f_val = open('valid.phn', 'w')
    f_test = open('test.phn', 'w')
    for el in train['phonemes']:
        el = clear_phones(el)
        f_train.write(el + '\n')
    for i in val['phonemes']:
        i = clear_phones(i)
        f_val.write(i + '\n')
    for j in test['phonemes']:
        j = clear_phones(j)
        f_test.write(j + '\n')


def sort_files(csv_original, csv_created):
    csv_init = pd.read_csv(csv_original, sep='\t')
    csv_to_sort = pd.read_csv(csv_created, sep='\t')
    name_files = csv_init['path'].tolist()
    a = []
    b = []
    for el in name_files:
        index = csv_to_sort.index[csv_to_sort['/data/user/m/cmacaire/unsupervised_wav2vec/na/no_silence/valid/clips_na_16khz'] == el].tolist()[0]
        row = csv_to_sort.iloc[index].tolist()
        a.append(row[0])
        b.append(row[1])
    df_results = pd.DataFrame({'/data/user/m/cmacaire/unsupervised_wav2vec/na/no_silence/valid/clips_na_16khz': a, '': b})
    df_results.to_csv('df_sort_valid.tsv', sep='\t', index=False)


if __name__ == '__main__':
    train, valid, test = load_and_process('/home/cmacaire/train.tsv',
                                          '/home/cmacaire/val.tsv',
                                          '/home/cmacaire/test.tsv')
    create_word_files(train, valid, test)
    create_letter_files(train, valid, test)
    # create_phones_files(train, valid, test)
    # sort_files('/home/cmacaire/Desktop/unsupervised-wav2vec/na/val_original.tsv', '/home/cmacaire/Desktop/unsupervised-wav2vec/na/new_tsv/valid.tsv')