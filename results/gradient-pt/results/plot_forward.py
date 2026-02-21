import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from scipy.signal import savgol_filter

# spp_list = [4, 8, 16, 32, 48]

def plot(input_file, label, xlabel, ylabel, output_file):
    sns.set_theme(style='darkgrid')
    all_data = np.loadtxt(input_file)

    x = np.linspace(0, all_data.shape[0] + 1, all_data.shape[0])

    dfs = pd.DataFrame({'label': label, 'x': x, 'y': np.log(all_data)})
    f, ax = plt.subplots()
    ax.plot(dfs['x'], dfs['y'], label=dfs['label'])
    # sns.lineplot(data=dfs, x='x', y='y', hue='label', ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.legend()
    f.savefig(f'{output_file}', bbox_inches='tight', pad_inches=0.2)

def plot_diff(single_input_file, multi_input_file, xlabel, ylabel, output_file):
    sns.set_theme(style='darkgrid')
    single_data = np.loadtxt(single_input_file)
    multi_data = np.loadtxt(multi_input_file)

    x = np.linspace(0, single_data.shape[0], single_data.shape[0])

    single_dfs = pd.DataFrame({'label': 'Single ReSTIR', 'x': x, 'y': single_data})
    multi_dfs = pd.DataFrame({'label': 'Multi ReSTIR', 'x': x, 'y': multi_data})

    f, ax = plt.subplots()
    dfs = pd.concat([single_dfs, multi_dfs])
    sns.lineplot(data=dfs, x='x', y='y', hue='label', ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    f.savefig(f'{output_file}')

def plotdiff_with_arg(args):
    plot_diff(args.single, args.multi, args.xlabel,
              args.ylabel, args.output)

def plot_with_arg(args):
    file_withour_ext = os.path.splitext(args.input)[0]
    plot(args.input, label=args.label,
         xlabel=args.xlabel, ylabel=args.ylabel,
         output_file=f'{file_withour_ext}_data.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data visualization')
    subparsers = parser.add_subparsers()

    parser_plot = subparsers.add_parser('plot', help='Plot data for a single method.')
    parser_plot.add_argument('--input', type=str, help='Path to the input file.')
    parser_plot.add_argument('--label', type=str, help='Label for the plot.')
    parser_plot.add_argument('--xlabel', type=str, default='# Frame', help='Label for the x-axis.')
    parser_plot.add_argument('--ylabel', type=str, default='RMSE (log scale)', help='Label for the x-axis.')
    parser_plot.set_defaults(func=plot_with_arg)

    parser_plotdiff = subparsers.add_parser('plotdiff', help='Plot data for comparison.')
    parser_plotdiff.add_argument('--single', type=str, help='Path to the single input file.')
    parser_plotdiff.add_argument('--multi', type=str, help='Path to the multi input file.')
    parser_plotdiff.add_argument('--xlabel', type=str, default='# Frame', help='Label for the x-axis.')
    parser_plotdiff.add_argument('--ylabel', type=str, default='RMSE', help='Label for the x-axis.')
    parser_plotdiff.add_argument('--output', type=str, help='Path to the output file.')
    parser_plotdiff.set_defaults(func=plotdiff_with_arg)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
