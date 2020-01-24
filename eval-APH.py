#!/usr/bin/env python3
"""Evaluate APH for LCNN
Usage:
    eval-APH.py <src> <dst>
    eval-APH.py (-h | --help )

Examples:
    ./eval-APH.py post/RUN-ITERATION/0_010 post/RUN-ITERATION/0_010-APH

Arguments:
    <src>                Source directory that stores preprocessed npz
    <dst>                Temporary output directory

Options:
   -h --help             Show this screen.
"""

import os
import glob
import os.path as osp
import subprocess

import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
from docopt import docopt

mpl.rcParams.update({"font.size": 18})
plt.rcParams["font.family"] = "Times New Roman"
del mpl.font_manager.weight_dict["roman"]
mpl.font_manager._rebuild()

image_path = "data/wireframe/valid-images/"
line_gt_path = "data/wireframe/valid/"
output_size = 128


def main():
    args = docopt(__doc__)
    src_dir = args["<src>"]
    tar_dir = args["<dst>"]

    output_file = osp.join(tar_dir, "result.mat")
    target_dir = osp.join(tar_dir, "mat")
    os.makedirs(target_dir, exist_ok=True)
    print(f"intermediate matlab results will be saved at: {target_dir}")

    file_list = glob.glob(osp.join(src_dir, "*.npz"))
    thresh = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]
    for t in thresh:
        for fname in file_list:
            name = fname.split("/")[-1].split(".")[0]
            mat_name = name + ".mat"
            npz = np.load(fname)
            lines = npz["lines"].reshape(-1, 4)
            scores = npz["score"]
            for j in range(len(scores) - 1):
                if scores[j + 1] == scores[0]:
                    lines = lines[: j + 1]
                    scores = scores[: j + 1]
                    break
            idx = np.where(scores > t)[0]
            os.makedirs(osp.join(target_dir, str(t)), exist_ok=True)
            sio.savemat(osp.join(target_dir, str(t), mat_name), {"lines": lines[idx]})

    cmd = "matlab -nodisplay -nodesktop "
    cmd += '-r "dbstop if error; '
    cmd += "eval_release('{:s}', '{:s}', '{:s}', '{:s}', {:d}); quit;\"".format(
        image_path, line_gt_path, output_file, target_dir, output_size
    )
    print("Running:\n{}".format(cmd))
    os.environ["MATLABPATH"] = "matlab/"
    subprocess.call(cmd, shell=True)

    mat = sio.loadmat(output_file)
    tps = mat["sumtp"]
    fps = mat["sumfp"]
    N = mat["sumgt"]
    rcs = sorted(list((tps / N)[:, 0]))
    prs = sorted(list((tps / np.maximum(tps + fps, 1e-9))[:, 0]))[::-1]

    print(
        "f measure is: ",
        (2 * np.array(prs) * np.array(rcs) / (np.array(prs) + np.array(rcs))).max(),
    )

    recall = np.concatenate(([0.0], rcs, [1.0]))
    precision = np.concatenate(([0.0], prs, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    print("AP is: ", np.sum((recall[i + 1] - recall[i]) * precision[i + 1]))

    f = interpolate.interp1d(rcs, prs, kind="cubic", bounds_error=False)
    x = np.arange(0, 1, 0.01) * rcs[-1]
    y = f(x)
    plt.plot(x, y, linewidth=3, label="L-CNN")

    f_scores = np.linspace(0.2, 0.8, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color="green", alpha=0.3)
        plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4)

    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.yticks(np.arange(0, 1.0, step=0.1))
    plt.legend(loc=3)
    plt.title("PR Curve for APH")
    plt.savefig("apH.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("apH.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plt.tight_layout()
    main()
