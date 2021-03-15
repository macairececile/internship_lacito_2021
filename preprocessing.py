import xml.etree.ElementTree as et
from os import listdir
from os.path import join, isfile
import sox
from pathlib import Path
import csv
import pandas as pd
import argparse


def get_files_from_directory(path):
    """
    Get all files from directory
    :param path: path where transcripts + wav files are stored
    :return: files
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files


def extract_information(xml_file):
    """
    Extract the transcription timecodes and ID from an xml file at the sentence level
    :param xml_file: transcription file
    :return: information (start audio, end audio, sentence id)
    """
    information = {}
    tree = et.parse(xml_file)
    root = tree.getroot()  # lexical resources
    sentences = root.findall('S')

    for child in sentences:
        id = child.attrib.get('id')
        transcript = child.find('FORM').text
        timecode = child.find('AUDIO').attrib
        info = [timecode['start'], timecode['end'], id]
        information[transcript] = info

    return information


def create_audio_tsv(args):
    """
    Create audios at the sentence level and create a tsv file which links each new audio file with the corresponding sentence
    :param args: path
    """
    path = args.path
    files = get_files_from_directory(path+'trans/')

    tsv = open(path + 'all.tsv', 'wt')
    tsv_writer = csv.writer(tsv, delimiter='\t')
    tsv_writer.writerow(['path', 'sentence'])

    for f in files:
        info = extract_information(path + 'trans/' + f)

        wav_dir = Path(path) / "wav" / "process_wav/"
        wav_dir.mkdir(exist_ok=True, parents=True)

        wav_file = path + 'wav/' + f[:-4] + '.wav'

        for k, v in info.items():
            tfm = sox.Transformer()
            tfm.trim(float(v[0])-0.2, float(v[1])+0.2)
            tfm.compand()
            output_wav = f[:-4] + '_' + v[2] + '.wav'
            tfm.build_file(wav_file, str(wav_dir) + '/' + output_wav)
            tsv_writer.writerow([output_wav, k])

    tsv.close()


def create_dataset(args):
    """
    Create train/val/test tsv files (ratio 70 / 15 / 15)
    :param args: path
    """
    path = args.path

    corpus = pd.read_csv(path + 'all.tsv', sep='\t')
    corpus.sample(frac=1)

    size_corpus = corpus.shape[0]

    split = [int(size_corpus * 0.7), int(size_corpus * 0.15)]

    train = corpus.iloc[:split[0]]
    val = corpus.iloc[split[0]:split[0] + split[1]]
    test = corpus.iloc[split[0] + split[1]:]

    train.to_csv(path + 'train.tsv', index=False, sep='\t')
    val.to_csv(path + 'val.tsv', index=False, sep='\t')
    test.to_csv(path + 'test.tsv', index=False, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    create_audio = subparsers.add_parser("create_audio",
                                         help="Create audio per sentences from xml and wav and store the info in a tsv file. "
                                              "Make sure the transcriptions are in /trans and wav in /wav")
    create_audio.add_argument('--path', type=str, required=True, help="path of the corpus with wav and transcription files.")
    create_audio.set_defaults(func=create_audio_tsv)

    split_dataset = subparsers.add_parser("create_dataset",
                                          help="Create dataset - train/val/test tsv files.")
    split_dataset.add_argument('--path', required=True, help="path of the corpus with wav and transcription files")
    split_dataset.set_defaults(func=create_dataset)

    args = parser.parse_args()
    args.func(args)
