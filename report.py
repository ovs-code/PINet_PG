import ast
import os
import math

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from options.infer_options import InferOptions
from pipeline import InferencePipeline

A4 = dict(figsize=(8.27, 11.69), dpi=100)
LOSS_GROUPS = [
    ['pair_GANloss', 'PB_fake', 'PB_real', 'PP_fake', 'PP_real'],
    ['L1_plus_perceptualLoss', 'L1', 'percetual'],
    ['parsing1']
]
HYPERPARAMETERS = [
    'name', 'lr', 'batchSize',
    'lambda_A', 'lambda_B', 'lambda_GAN',
    'sepiter', 'niter', 'niter_decay'
]

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
                continue
            for c in '(),':
                line = line.replace(c, '')
            epochs.append(parse_line(line))
    log = pd.DataFrame(epochs)
    del log['iters'], log['time']
    return log.groupby('epoch').mean()


def parse_val_loss(logfile):
    epochs = []
    with open(logfile) as f:
        for line in f:
            if not line.startswith('Validation Loss:'):
                continue  # to the next line
            line = line.lstrip('Validation Loss: ')
            epochs.append(ast.literal_eval(line))

    log = pd.DataFrame(epochs, index=range(1, len(epochs)+1))
    log.index.name = 'epoch'
    return log


def plot_row(row, labels, data):
    upper = max(df[label].max() for label in labels for df in data if label in df)
    lower = min(df[label].min() for label in labels for df in data if label in df)
    for ax, df in zip(row, data):
        for label in labels:
            if label in df:
                ax.plot(df.index, df[label], label=label)
        ax.set_ylim(lower - 0.1, upper + 0.1)
    handles, labels = ax.get_legend_handles_labels()
    leg = row[1].legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc=0)
    for line in leg.get_lines():
        line.set_linewidth(4)


def make_report(checkpoint_dir):
    # load models / data
    logfile = os.path.join(checkpoint_dir, 'loss_log.txt')
    optfile = os.path.join(checkpoint_dir, 'opt.txt')
    opts = parse_opts(optfile)
    train_loss = parse_train_loss(logfile)
    val_loss = parse_val_loss(logfile)
    pipeline_opts = InferOptions().parse([
            '--name',
            opts['name'],
            '--checkpoints_dir',
            checkpoint_dir[: checkpoint_dir.rfind('/')],
        ] + (['--remove_background'] if opts['remove_background'] == 'True' else []))
    pipeline = InferencePipeline.from_opts(pipeline_opts)

    with open('test_data/test.lst') as f:
        persons = [line.strip() for line in f]

    # generate pdf report from data
    with PdfPages(f'reports/{opts["name"]}.pdf') as pdf:
        # page 1: table of hyperparameters
        fig, ax = plt.subplots(**A4)
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ptable = ax.table(
            cellText=[[label, opts.get(label, '')] for label in HYPERPARAMETERS],
            colLabels=['parameter', 'value'],
            loc='center'
        )
        ptable.set_fontsize(18)
        ptable.scale(1, 2)

        pdf.savefig()
        plt.close()
        
        # page 2: loss graphs
        fig, axs = plt.subplots(3, 2, **A4)
        axs[0, 0].set_title('Training Loss')
        axs[0, 1].set_title('Validation Loss')

        for row, group in zip(axs, LOSS_GROUPS):
            plot_row(row, group, (train_loss, val_loss))

        fig.subplots_adjust(right=0.8)

        pdf.savefig()
        plt.close()
        
        # page 3-n: grids of test examples
        for j in range(math.ceil(len(persons) / 4)):
            group = persons[j*4: (j+1)*4]

            fig, axs = plt.subplots(5, 5, **A4)
            [axi.axis('off') for axi in axs.ravel()]

            for i, person in enumerate(group):
                source = Image.open(f'test_data/test/{person}.jpg')
                axs[0, i+1].imshow(source)
                axs[i+1, 0].imshow(source)
                for j, t in enumerate(group):
                    target = Image.open(f'test_data/test/{t}.jpg')
                    out = pipeline.map_to(source, target)
                    axs[i+1, j+1].imshow(out)
            fig.tight_layout()
            pdf.savefig()
            plt.close()

        


if __name__ == '__main__':
    import sys
    make_report(sys.argv[1])
