#! /usr/bin/env python

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Alexander Rush, 2017.

""" Generate beam search visualization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import json
import shutil
from string import Template

import networkx as nx
from networkx.readwrite import json_graph

PARSER = argparse.ArgumentParser(
    description="Generate beam search visualizations")
PARSER.add_argument(
    "-d", "--data", type=str, required=True,
    help="path to the beam search data file")
PARSER.add_argument(
    "-o", "--output_dir", type=str, required=True,
    help="path to the output directory")
ARGS = PARSER.parse_args()

HTML_TEMPLATE = Template("""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Beam Search</title>
    <link rel="stylesheet" type="text/css" href="tree.css">
    <script src="http://d3js.org/d3.v3.min.js"></script>
  </head>
  <body>
    <script>
      var treeData = $DATA
    </script>
    <script src="tree.js"></script>
  </body>
</html>""")


def create_graph(predicted_ids, scores, names, total_score):
    levels = [i for i in range(1, len(predicted_ids))]
    len_seq = len(predicted_ids[0])
    G = nx.DiGraph()
    G.add_node((0, 0))
    G.nodes[(0, 0)]["name"] = "ROOT"
    for i, j in enumerate(predicted_ids[0]):
        parent_node = (0, i)
        new_node = (0, i + 1)
        score_str = '%.3f' % float(scores[0][i]) if scores[0][i] is not None else '-inf'
        name = names[0][i]
        G.add_node(new_node, score=score_str, name=name)
        G.add_edge(parent_node, new_node)
    G.add_node((0, len_seq+1), score=total_score[0], name="TOTAL SCORE")
    G.add_edge((0, len_seq), (0, len_seq+ 1))
    for l in levels:
        ind = 0
        new_level = False
        print(len(predicted_ids[l]))
        for i, j in enumerate(predicted_ids[l]):
            if names[l][i] != G.nodes[(0, i+1)]["name"] and new_level == False:
                print(i)
                print(names[l][i])
                print(G.nodes[(0, i+1)]["name"])
                score_str = '%.3f' % float(scores[l][i]) if scores[l][i] is not None else '-inf'
                name = names[l][i]
                G.add_node((l, ind), score=score_str, name=name)
                G.add_edge((0, i), (l, ind))
                ind += 1
                new_level = True
                continue
            if new_level:
                score_str = '%.3f' % float(scores[l][i]) if scores[l][i] is not None else '-inf'
                name = names[l][i]
                G.add_node((l, ind), score=score_str, name=name)
                G.add_edge((l, ind - 1), (l, ind))
                ind += 1
        G.add_node((l, ind), score=total_score[l], name="TOTAL SCORE")
        G.add_edge((l, ind - 1), (l, ind))
    return G


def main():
    beam_data = json.load(open(ARGS.data, 'r'))

    # Optionally load vocabulary data
    vocab = None

    if not os.path.exists(ARGS.output_dir):
        os.makedirs(ARGS.output_dir)

    path_base = os.path.dirname(os.path.realpath(__file__))

    # Copy required files
    shutil.copy2(path_base + "/beam_search_viz/tree.css", ARGS.output_dir)
    shutil.copy2(path_base + "/beam_search_viz/tree.js", ARGS.output_dir)

    graph = create_graph(
        predicted_ids=beam_data["predicted_ids"],
        scores=beam_data["scores"],
        names=beam_data["names"],
        total_score=beam_data["total_score"])

    json_str = json.dumps(
        json_graph.tree_data(graph, root=(0, 0)),
        ensure_ascii=True)

    html_str = HTML_TEMPLATE.substitute(DATA=json_str)
    output_path = os.path.join(ARGS.output_dir, ARGS.data[:-5].split('/')[-1]+".html")
    with open(output_path, "w") as file:
        file.write(html_str)
    print(output_path)


if __name__ == "__main__":
    main()
