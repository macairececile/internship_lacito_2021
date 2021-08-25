# ----------- Libraries ----------- #
import argparse
import subprocess
import preprocessing_text_na as prep
import preprocessing_text_japhug as prep_jap
from datasets import load_dataset


def load_data(file):
    dataset = load_dataset('csv', data_files=[file], delimiter='\t')
    dataset = dataset['train']
    return dataset


def preprocessing_data_LM(file, lang, output):
    """Get the train text sentences and store it in a .txt file"""
    data = load_data(file)
    if lang == "na":
        data = data.map(prep.final_text_words)
    elif lang == "japhug":
        data = data.map(prep_jap.final_text_words)
    with open(output + 'train_kenLM.txt', 'w') as f:
        for el in data['sentence']:
            f.write(el + '\n')


def create_language_model(arguments):
    """Create 2-, 3-, and 4-gram kenLM language model"""
    preprocessing_data_LM(arguments.data, arguments.lang, arguments.output_path)
    for i in range(2, 5):
        command = './kenlm/build/bin/lmplz -o ' + str(i) + ' < ' + arguments.output_path + 'train_kenLM.txt > ' + arguments.output_path + arguments.lang + '.arpa'
        subprocess.check_output(command, shell=True)


# ----------- Arguments ----------- #
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')

language_m = subparsers.add_parser("run",
                                   help="Create a kenLM language model.")
language_m.add_argument('--data', type=str, required=True,
                        help="Data file.")
language_m.add_argument('--output_path', type=str, required=True,
                        help="Name of the file to store language model data.")
language_m.add_argument('--lang', type=str, required=True, choices={"japhug", "na"},
                        help="Language of the corpus.")
language_m.set_defaults(func=create_language_model)

arguments = parser.parse_args()
arguments.func(arguments)
