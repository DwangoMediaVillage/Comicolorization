"""
python3 bin/make_graph_report.py \
    projectA \
    projectB \
    --path_output `pwd`/graph_linelike_without_real.png \
    --key_x iteration \
    --key_y validation/main/loss validation/train/main/loss \
    --title loss-graph\
    --figsize 16 9 \
    --xlim 0 100000 \
    --ylim 0 500 \
    --filter_k_size 5 \
"""

import argparse
import json
import os
from typing import List, Dict

import matplotlib
import matplotlib.font_manager as fm
import numpy
import scipy.signal

parser = argparse.ArgumentParser()
parser.add_argument("paths_result_directory", nargs='+')
parser.add_argument("--path_root", type=str)
parser.add_argument("--path_output", required=False, type=str, help="when not specified, show figure")
parser.add_argument("--key_x", required=True, type=str, help="key name of values for x axis")
parser.add_argument("--key_y", nargs='+', type=str, help="key name of values for y axis")

parser.add_argument('--title', type=str)
parser.add_argument('--project_name', nargs='+', help="can set name of each projects")
parser.add_argument('--font_path', type=str)
parser.add_argument("--figsize", nargs='+', type=float, default=[16, 9])
parser.add_argument("--xlim", nargs='+', type=float)
parser.add_argument("--ylim", nargs='+', type=float)
parser.add_argument("--filter_k_size", type=int, default=1)
parser.add_argument("--method_filter", type=str, default='mean', help="mean or median")
parser.add_argument("--line_gradient", choices=['auto', 'gray', 'dashed', 'width', 'color'], default='auto', help="gray=暗さ｜dashed=点線の間隔｜width=太さ｜color=色")
args = parser.parse_args()

show_figure = (args.path_output is None)

paths_result_directory = args.paths_result_directory
if args.path_root is not None:
    paths_result_directory = [os.path.join(args.path_root, relpath) for relpath in paths_result_directory]

if args.project_name is not None:
    project_names = args.project_name
else:
    project_names = [os.path.split(path_result)[1] for path_result in paths_result_directory]

paths_logreport = [os.path.join(path_result, 'log.txt') for path_result in paths_result_directory]

num_projcet = len(project_names)


def pass_filter(y):
    if args.method_filter == 'mean':
        y = numpy.pad(y, (args.filter_k_size // 2,), 'edge')
        filter = numpy.ones(args.filter_k_size) / args.filter_k_size
        return numpy.convolve(y, filter, 'valid')
    elif args.method_filter == 'median':
        return scipy.signal.medfilt(y, args.filter_k_size)
    else:
        raise Exception("{} is not defined".format(args.method_filter))


# prepare figure
if not show_figure:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

font = fm.FontProperties(fname=args.font_path)

figure = plt.figure(figsize=args.figsize)
ax = figure.add_subplot(111)

colormap = plt.get_cmap('Dark2', 8)
colors = colormap(range(8))

line_style_list = ['-', '--', '-.', ':']

# draw with each report
for (project_name, path_logreport, color) in zip(project_names, paths_logreport, colors):
    assert os.path.exists(path_logreport), "there is no file at {}".format(path_logreport)

    # log_list is like : [ { "a": a1, "b": b1 }, { "a": a2, "b": b2 }, ... ]
    log_list = json.load(open(path_logreport))  # type: List[Dict]
    keys = log_list[0].keys()

    # log_table is like : { "a": [a1, a2, ...], "b": [b1, b2, ...] }
    log_table = {}
    for key in keys:
        values_with_key = numpy.array([log[key] for log in log_list])
        log_table[key] = values_with_key

    # make figure
    key_x = args.key_x
    keys_y = args.key_y  # type: List[str]

    assert key_x in keys, "there is no key('{0}') in log report keys('{1}')".format(key_x, ', '.join(keys))
    keys_y = [key_y for key_y in keys_y if key_y in keys]

    if args.line_gradient == 'auto':
        if len(keys_y) > 2:
            if num_projcet > 1:
                line_gradient = 'dashed'
            else:
                line_gradient = 'color'
        else:
            line_gradient = 'gray'
    else:
        line_gradient = args.line_gradient

    x = log_table[key_x]
    i = 0

    for i, key_y in enumerate(keys_y):
        y = log_table[key_y]
        y = pass_filter(y)
        label = '[{0}] {1}'.format(project_name, key_y)
        if line_gradient == 'gray':
            ax.plot(x, y, label=label, color=color / (i + 1))
        elif line_gradient == 'dashed':
            ax.plot(x, y, label=label, color=color, linestyle=line_style_list[i])
        elif line_gradient == 'width':
            ax.plot(x, y, label=label, color=color, linewidth=(len(keys_y) + 1) / 2 - i / 2)
        if line_gradient == 'color':
            ax.plot(x, y, label=label, color=colors[i])

if args.title is not None:
    ax.set_title(args.title, fontproperties=font)

if args.xlim is not None:
    ax.set_xlim(args.xlim)
if args.ylim is not None:
    ax.set_ylim(args.ylim)

ax.set_xlabel(args.key_x)
ax.set_ylabel(args.key_y)

ax.legend(prop=font)

if show_figure:
    plt.show(block=True)
else:
    figure.savefig(args.path_output)
