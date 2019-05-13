#!/usr/bin/env python3
import os
import glob
import os.path as osp

import cv2
import numpy as np
import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt

from lcnn.utils import parmap

GT = "data/wireframe/valid/*.npz"
WF = "/data/lcnn/wirebase/result/wireframe/wireframe_1_rerun-baseline_0.5_0.5/2/*.mat"
AFM = "/data/lcnn/wirebase/result/wireframe/afm/*.npz"
IMGS = "/data/lcnn/wirebase/Wireframe/v1.1/test/*.jpg"


def imshow(im):
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
    gts = sorted(glob.glob(GT))
    afm = sorted(glob.glob(AFM))
    wf = sorted(glob.glob(WF))
    img = sorted(glob.glob(IMGS))

    prefix = "/data/lcnn/wirebase/myplot/"
    os.makedirs(osp.join(prefix, "GT"), exist_ok=True)
    os.makedirs(osp.join(prefix, "LSD"), exist_ok=True)
    os.makedirs(osp.join(prefix, "AFM"), exist_ok=True)
    os.makedirs(osp.join(prefix, "WF"), exist_ok=True)
    os.makedirs(osp.join(prefix, "LL"), exist_ok=True)

    def draw(args):
        i, (wf_name, gt_name, afm_name, img_name) = args
        img = cv2.imread(img_name, 0)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        lsd_line, _, _, lsd_score = lsd.detect(img)
        lsd_line = lsd_line.reshape(-1, 2, 2)
        lsd_score = lsd_score.flatten()
        img = cv2.imread(img_name)[:, :, ::-1]

        with np.load(gt_name) as fgt:
            gt_line = fgt["lpos"][:, :, :2]
            gt_line[:, :, 0] *= img.shape[0] / 128
            gt_line[:, :, 1] *= img.shape[1] / 128

        with np.load(afm_name) as fafm:
            afm_line = fafm["lines"].reshape(-1, 2, 2)[:, :, ::-1]

        wf_line = scipy.io.loadmat(wf_name)["lines"].reshape(-1, 2, 2)
        wf_line = wf_line[:, :, ::-1]

        plt.figure("GT")
        imshow(img)
        for a, b in gt_line - 0.5:
            plt.plot([a[1], b[1]], [a[0], b[0]], color="orange", linewidth=0.5)
            plt.scatter(a[1], a[0], color="#33FFFF", s=1.2, edgecolors="none", zorder=5)
            plt.scatter(b[1], b[0], color="#33FFFF", s=1.2, edgecolors="none", zorder=5)
        plt.savefig(osp.join(prefix, "GT", f"{i:05}"), dpi=3000, bbox_inches=0)
        plt.close()

        plt.figure("LSD")
        imshow(img)
        for a, b in lsd_line[:, :, ::-1] - 0.5:
            plt.plot([a[1], b[1]], [a[0], b[0]], color="orange", linewidth=0.5)
            plt.scatter(a[1], a[0], color="#33FFFF", s=1.2, edgecolors="none", zorder=5)
            plt.scatter(b[1], b[0], color="#33FFFF", s=1.2, edgecolors="none", zorder=5)
        plt.savefig(osp.join(prefix, "LSD", f"{i:05}"), dpi=3000, bbox_inches=0)
        plt.close()

        plt.figure("AFM")
        imshow(img)
        for a, b in afm_line - 0.5:
            plt.plot([a[1], b[1]], [a[0], b[0]], color="orange", linewidth=0.5)
            plt.scatter(a[1], a[0], color="#33FFFF", s=1.2, edgecolors="none", zorder=5)
            plt.scatter(b[1], b[0], color="#33FFFF", s=1.2, edgecolors="none", zorder=5)
        plt.savefig(osp.join(prefix, "AFM", f"{i:05}"), dpi=3000, bbox_inches=0)
        plt.close()

        plt.figure("WF")
        imshow(img)
        for a, b in wf_line - 0.5:
            plt.plot([a[1], b[1]], [a[0], b[0]], color="orange", linewidth=0.5)
            plt.scatter(a[1], a[0], color="#33FFFF", s=1.2, edgecolors="none", zorder=5)
            plt.scatter(b[1], b[0], color="#33FFFF", s=1.2, edgecolors="none", zorder=5)
        plt.savefig(osp.join(prefix, "WF", f"{i:05}"), dpi=3000, bbox_inches=0)
        plt.close()

    parmap(draw, enumerate(zip(wf, gts, afm, img)))


if __name__ == "__main__":
    main()
