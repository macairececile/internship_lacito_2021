import argparse


# Create a text file with the .tex format which makes a parallel between HYP et HYP with LM
# output file containing:
# Ref: & tʰi˩˥ le˧qæ˧˥ mv̩˩mɤ\hl{˩}tʰɑ˥dʑo˩ tʰi˩˥ hĩ˧ɳɯ˩ le˧wo˧˥ ə˧mv̩˧ki˥ \\
# Hyp: & tʰi˩˥ le˧qæ˧˥ mv̩˩ mɤ\hl{˧}tʰɑ˥dʑo˩ tʰi˩˥ hĩ˧ɳɯ˩ le˧wo˧˥ ə˧mv̩˧ki˥ \\
# Ref: & tʰi˩˥ le˧qæ˧˥ mv̩˩mɤ\hl{˩}tʰɑ˥dʑo˩ tʰi˩˥ hĩ˧ɳɯ˩ le˧wo˧\hl{˥} ə˧\hl{m}v̩˧ki˥ \\
# Hyp\_LM: & tʰi˩˥ le˧qæ˧˥ mv̩˩ mɤ\hl{˧}tʰɑ\hl{˧}˥ dʑo˩ tʰi˩˥ hĩ˧ɳɯ˩ le˧wo˧ ə˧v̩˧ki˥ \\

def create_latex_lm(arguments):
    c_final = []
    with open(arguments.latex_no_lm, 'r') as f1:
        content = f1.readlines()
    with open(arguments.latex_lm, 'r') as f2:
        content2 = f2.readlines()
    for i in range(0, len(content), 3):
        c_final.append(content[i])
        c_final.append(content[i + 1])
        c_final.append(content2[i])
        c_final.append(content2[i + 1])
        c_final.append(content[i + 2])
    with open('latex_final_lm.txt', 'a') as final_f:
        for el in c_final:
            final_f.write(el)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')

    latex = subparsers.add_parser("create",
                                  help="Create a latex file with results from lm and from no lm.")
    latex.add_argument('--latex_no_lm', type=str, required=True,
                       help="Txt file with results not using LM.")
    latex.add_argument('--latex_lm', type=str, required=True,
                       help="Txt file with results using LM.")
    latex.set_defaults(func=create_latex_lm)

    args = parser.parse_args()
    args.func(args)
