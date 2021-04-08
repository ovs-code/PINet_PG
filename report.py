import ast
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from options.infer_options import InferOptions
from pipeline import InferencePipeline
from tool import reorder_pose


def parse_opts(optfile):
    result = dict()
    with open(optfile) as f:
        for line in f:
            if ': ' not in line:
                continue
            key, value = line.split(': ')
            result[key] = value.strip()
    return result


def parse_line(line):
    items = line.split(' ')
    keys = [key.strip(':') for key in items[::2]]
    values = [ast.literal_eval(value) for value in items[1::2]]
    return dict(zip(keys, values))


def parse_train_loss(logfile):
    epochs = []
    with open(logfile) as f:
        for line in f:
            if not line.startswith('(epoch:'):
                continue  # to the next line
            for c in '(),':
                line = line.replace(c, '')
            epochs.append(parse_line(line))
    log = pd.DataFrame(epochs)
    del log['iters'], log['time']
    if 'cycle_loss' in log:
        del log['cycle_loss']
    return log.groupby('epoch').mean()


def parse_val_loss(logfile):
    epochs = []
    with open(logfile) as f:
        for line in f:
            if not line.startswith('Validation Loss:'):
                continue # to the next line
            line = line.lstrip('Validation Loss: ')
            epochs.append(ast.literal_eval(line))

    log = pd.DataFrame(epochs, index=range(1, len(epochs)+1))
    log.index.name = 'epoch'
    if 'cycle_loss' in log:
        del log['cycle_loss']
    return log


def plot_row(row, fig, labels, data):
    upper = max(df[label].max() for label in labels for df in data)
    lower = min(df[label].min() for label in labels for df in data)
    for ax, df in zip(row, data):
        for label in labels:
            ax.plot(df.index, df[label], label=label)
        ax.set_ylim(lower - 0.1, upper + 0.1)
    handles, labels = ax.get_legend_handles_labels()
    leg = row[1].legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc=0)
    for line in leg.get_lines():
        line.set_linewidth(4)


def make_report(checkpoint_dir):
    logfile = os.path.join(checkpoint_dir, 'loss_log.txt')
    optfile = os.path.join(checkpoint_dir, 'opt.txt')
    opts = parse_opts(optfile)
    train_loss = parse_train_loss(logfile)
    val_loss = parse_val_loss(logfile)

    with PdfPages('reports/' + opts['name'] + '.pdf') as pdf:
        labels = [
            'name', 'lr', 'batchSize',
            'lambda_A', 'lambda_B', 'lambda_GAN',
            'sepiter', 'niter', 'niter_decay'
        ]
        fig, ax = plt.subplots()

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ptable = ax.table(
            cellText=[[label, opts.get(label, '')] for label in labels],
            colLabels=['parameter', 'value'],
            loc='center'
        )
        ptable.set_fontsize(18)
        ptable.scale(1, 2)

        pdf.savefig()
        plt.close()

        fig, axs = plt.subplots(5, 5, figsize=(8.27, 11.69), dpi=100)
        for row in axs:
            for ax in row:
                ax.axis('off')
        with open('test_data/test.lst') as f:
            persons = [line.strip() for line in f]
        pip_opts = InferOptions().parse(['--name', 'fashion_PInet_cycle'])
        pipeline = InferencePipeline.from_opts(pip_opts)

        for i, person in enumerate(persons):
            image = Image.open(f'test_data/test/{person}.jpg')
            axs[0, i+1].imshow(image)
            axs[i+1, 0].imshow(image)
            pipeline.segmentator.path = f'test_data/testSPL2/{person}.png'
            for j, target in enumerate(persons):
                target_pose = np.load(f'test_data/testK/{target}.jpg.npy')
                out = pipeline(image, reorder_pose(target_pose))
                axs[i+1, j+1].imshow(out)
        fig.tight_layout()
        pdf.savefig()
        plt.close()

        fig, axs = plt.subplots(3, 2, figsize=(8.27, 11.69), dpi=100)
        for row in axs:
            for ax in row:
                ax.set_ylim(0, 2.5)
        axs[0, 0].set_title('Training Loss')
        axs[0, 1].set_title('Validation Loss')
        plot_row(
            axs[0], fig,
            ['pair_GANloss', 'PB_fake', 'PB_real', 'PP_fake', 'PP_real'],
            (train_loss, val_loss)
        )
        plot_row(
            axs[1], fig,
            ['L1_plus_perceptualLoss', 'L1', 'percetual'],
            (train_loss, val_loss)
        )
        plot_row(axs[2], fig, ['parsing1'], (train_loss, val_loss))

        fig.subplots_adjust(right=0.8)
        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    import sys
    checkpoint_dir = sys.argv[1]
    make_report(checkpoint_dir)
