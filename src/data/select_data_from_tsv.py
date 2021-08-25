from os import listdir
from os.path import join, isfile, exists
import shutil
import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter


def get_files_from_directory(path):
    """
    Get all files from directory
    :param path: path where transcripts + wav files are stored
    :return: files
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files


def select_files(args):
    data = pd.read_csv(args.tsv_file, sep='\t')
    path = data['path']
    for p in path:
        shutil.copy(args.path_audio + p, args.output_path + p)
    xml_files = list(map(lambda i: '_'.join(i.split('_')[:-1]) + '.xml', path))
    xml_files = list(set(xml_files))
    for i in xml_files:
        if exists(args.path_xml + i):
            shutil.copy(args.path_xml + i, args.output_path + i)
        else:
            i = '_'.join(i.split('_')[:-1]) + '.xml'
            print(i)
            if exists(args.path_xml + i):
                shutil.copy(args.path_xml + i, args.output_path + i)
            else:
                i = '_'.join(i.split('_')[:-1]) + '.xml'
                print(i)
                shutil.copy(args.path_xml + i, args.output_path + i)


if __name__ == "__main__":
    parser = ArgumentParser(description="Select transcription files and audio clips according to the .tsv file.", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--tsv_file", required=True,
                        help="tsv file.")
    parser.add_argument("--path_audio", required=True,
                        help="Path of the audio files.")
    parser.add_argument("--output_path", required=True,
                        help="Path to store the files.")
    parser.add_argument("--path_xml", required=True,
                        help="Path of the xml files.")
    parser.set_defaults(func=select_files)
    args = parser.parse_args()
    args.func(args)