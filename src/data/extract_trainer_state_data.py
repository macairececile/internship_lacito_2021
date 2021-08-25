import json
from argparse import ArgumentParser, RawTextHelpFormatter


def extract_log_info(args):
    with open(args.log_file, "r") as f:
        res = ''
        data = json.load(f)
        log = data["log_history"]
        for i in range(1, len(log), 2):
            res += '(' + str(log[i]['epoch']) + ',' + str(log[i]['eval_cer']) + ') '
        print(res)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract the eval_cer score from the log file generated with the fine-tuning and print it.",
        formatter_class=RawTextHelpFormatter)

    parser.add_argument("--log_file", required=True,
                        help="Log file in json format.")
    parser.set_defaults(func=extract_log_info)
    args = parser.parse_args()
    args.func(args)
