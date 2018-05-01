import os
import sys
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str,
    default='output/model-id/data/model-iter')
parser.add_argument('--outdir', type=str,
    default='output/model-id/plot/plot-iter')
parser.add_argument('--num_sample', type=int, default=10)
parser.add_argument('--gpu_server', type=bool, default=True)
args = parser.parse_args()


if args.gpu_server:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

make_dirs(args.outdir)

files = os.listdir(args.indir)
files_dir = [f for f in files if os.path.isdir(os.path.join(args.indir, f))]
num_dir = len(files_dir)
dirnames = sorted(files_dir)

_num_dir = int(np.sqrt(num_dir))
for i in range(1, args.num_sample + 1):
    fig, axes = plt.subplots(nrows=_num_dir, ncols=_num_dir + 1,
                             figsize=(12, 12))
    for d, ax in zip(dirnames, axes.flat[:num_dir]):
        path = os.path.join(args.indir, d, '{}.png'.format(i))
        img = np.array(Image.open(path))
        ax.imshow(img)
        ax.set_title(d, fontsize=8)
        ax.set_axis_off()
    res_path = os.path.join(args.outdir, '{}.png'.format(i))
    print('Save ', res_path)
    plt.savefig(res_path, transparent=True)
    plt.clf()
    plt.close('all')