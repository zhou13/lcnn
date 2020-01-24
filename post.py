#!/usr/bin/env python3
"""Post-processing the output of neural network
Usage:
    post.py [options] <input-dir> <output-dir>
    post.py ( -h | --help )

Examples:
    post.py logs/logname/npz/000336000  result/logname

Arguments:
   input-dir                         Directory that stores the npz
   output-dir                        Output directory

Options:
   -h --help                         Show this screen.
   --plot                            Generate images besides npz files
   --thresholds=<thresholds>         A comma-separated list for thresholding
                                     [default: 0.006,0.010,0.015]
"""

import glob
import math
import os
import os.path as osp
import sys

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

from lcnn.postprocess import postprocess
from lcnn.utils import parmap

PLTOPTS = {"color": "#33FFFF", "s": 1.2, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.92, vmax=1.02)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def imshow(im):
    plt.close()
    sizes = im.shape
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, sizes[1] - 0.5])
    plt.ylim([sizes[0] - 0.5, -0.5])
    plt.imshow(im)


def main():
    args = docopt(__doc__)

    files = sorted(glob.glob(osp.join(args["<input-dir>"], "*.npz")))
    inames = sorted(glob.glob("data/wireframe/valid-images/*.jpg"))
    gts = sorted(glob.glob("data/wireframe/valid/*.npz"))
    prefix = args["<output-dir>"]

    inputs = list(zip(files, inames, gts))
    thresholds = list(map(float, args["--thresholds"].split(",")))

    def handle(allname):
        fname, iname, gtname = allname
        print("Processing", fname)
        im = cv2.imread(iname)
        with np.load(fname) as f:
            lines = f["lines"]
            scores = f["score"]
        with np.load(gtname) as f:
            gtlines = f["lpos"][:, :, :2]
        gtlines[:, :, 0] *= im.shape[0] / 128
        gtlines[:, :, 1] *= im.shape[1] / 128
        for i in range(1, len(lines)):
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        lines[:, :, 0] *= im.shape[0] / 128
        lines[:, :, 1] *= im.shape[1] / 128
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5

        for threshold in thresholds:
            nlines, nscores = postprocess(lines, scores, diag * threshold, 0, False)

            outdir = osp.join(prefix, f"{threshold:.3f}".replace(".", "_"))
            os.makedirs(outdir, exist_ok=True)
            npz_name = osp.join(outdir, osp.split(fname)[-1])

            if args["--plot"]:
                # plot gt
                imshow(im[:, :, ::-1])
                for (a, b) in gtlines:
                    plt.plot([a[1], b[1]], [a[0], b[0]], c="orange", linewidth=0.5)
                    plt.scatter(a[1], a[0], **PLTOPTS)
                    plt.scatter(b[1], b[0], **PLTOPTS)
                plt.savefig(npz_name.replace(".npz", ".png"), dpi=500, bbox_inches=0)

                thres = [0.96, 0.97, 0.98, 0.99]
                for i, t in enumerate(thres):
                    imshow(im[:, :, ::-1])
                    for (a, b), s in zip(nlines[nscores > t], nscores[nscores > t]):
                        plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=0.5)
                        plt.scatter(a[1], a[0], **PLTOPTS)
                        plt.scatter(b[1], b[0], **PLTOPTS)
                    plt.savefig(
                        npz_name.replace(".npz", f"_{i}.png"), dpi=500, bbox_inches=0
                    )

            nlines[:, :, 0] *= 128 / im.shape[0]
            nlines[:, :, 1] *= 128 / im.shape[1]
            np.savez_compressed(npz_name, lines=nlines, score=nscores)

    parmap(handle, inputs, 12)


if __name__ == "__main__":
    main()
