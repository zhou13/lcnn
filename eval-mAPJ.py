#!/usr/bin/env python3
"""Evaluate mAPJ for LCNN, AFM, and Wireframe
Usage:
    eval-mAPJ.py <path>...
    eval-mAPJ.py (-h | --help )

Examples:
    python eval-mAPJ.py logs/*

Arguments:
    <path>                           One or more directories that contain *.npz

Options:
   -h --help                         Show this screen.
"""

import os
import re
import glob
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.io import loadmat

import lcnn.models
from lcnn.metric import mAPJ, post_jheatmap

GT = "data/wireframe/valid/*.npz"
IM = "data/wireframe/valid-images/*.jpg"
WF = "/data/wirebase/result/junc/2/17"
AFM = "/data/wirebase/result/wireframe/afm/*.npz"

DIST = [0.5, 1.0, 2.0]


def evaluate_lcnn(im_list, gt_list, lcnn_list):
    # define result array to aggregate (n x 3) where 3 is (x, y, score)
    all_junc = np.zeros((0, 3))
    all_offset_junc = np.zeros((0, 3))
    # for each detected junction, which image they correspond to
    all_junc_ids = np.zeros(0, dtype=np.int32)
    # gt is a list since the variable gt number per image
    all_jc_gt = []

    for i, (lcnn_fn, gt_fn) in enumerate(zip(lcnn_list, gt_list)):
        with np.load(lcnn_fn) as npz:
            result = {name: arr for name, arr in npz.items()}
            jmap = result["jmap"]
            joff = result["joff"]

        with np.load(gt_fn) as npz:
            junc_gt = npz["junc"][:, :2]

        # for j in junc_gt:
        #     plt.scatter(round(j[1]), round(j[0]), c="red")
        # for j in juncs_wf:
        #     plt.scatter(round(j[1]), round(j[0]), c="blue")
        # plt.show()

        jun_c = post_jheatmap(jmap[0])
        all_junc = np.vstack((all_junc, jun_c))
        jun_o_c = post_jheatmap(jmap[0], offset=joff[0])
        all_offset_junc = np.vstack((all_offset_junc, jun_o_c))

        all_jc_gt.append(junc_gt)
        all_junc_ids = np.hstack((all_junc_ids, np.array([i] * len(jun_c))))

    # sometimes filter all and concat empty list will change dtype
    all_junc_ids = all_junc_ids.astype(np.int64)
    ap_jc = mAPJ(all_junc, all_jc_gt, DIST, all_junc_ids)
    ap_joc = mAPJ(all_offset_junc, all_jc_gt, DIST, all_junc_ids)
    print(f"  {ap_jc:.1f} | {ap_joc:.1f}")


def evaluate_wireframe(im_list, gt_list, juncs_wf):
    print("Compute WF mAP")
    juncs_wf = load_wf()
    all_junc = np.zeros((0, 3))
    all_junc_ids = np.zeros(0, dtype=np.int32)
    all_jc_gt = []
    for i, (im_fn, gt_fn, junc_wf) in enumerate(zip(im_list, gt_list, juncs_wf)):
        im = cv2.imread(im_fn)
        im = cv2.resize(im, (128, 128))

        with np.load(gt_fn) as npz:
            junc_gt = npz["junc"][:, :2]
        jun_c = sorted(junc_wf, key=lambda x: -x[2])[:1000]

        all_junc = np.vstack((all_junc, jun_c))
        all_jc_gt.append(junc_gt)
        all_junc_ids = np.hstack((all_junc_ids, np.array([i] * len(jun_c))))
    all_junc_ids = all_junc_ids.astype(np.int64)
    ap_jc = mAPJ(all_junc, all_jc_gt, DIST, all_junc_ids)
    print(f"  {ap_jc:.1f}")


def evaluate_afm(im_list, gt_list, afm):
    print("Compute AFM mAP")
    all_junc = np.zeros((0, 3))
    all_junc_ids = np.zeros(0, dtype=np.int32)
    all_jc_gt = []
    afm = glob.glob(AFM)
    afm.sort()
    for i, (im_fn, gt_fn, afm_fn) in enumerate(zip(im_list, gt_list, afm)):
        im = cv2.imread(im_fn)
        im = cv2.resize(im, (128, 128))

        with np.load(gt_fn) as npz:
            junc_gt = npz["junc"][:, :2]

        with np.load(afm_fn) as fafm:
            afm_line = fafm["lines"].reshape(-1, 2, 2)[:, :, ::-1]
            afm_score = -fafm["scores"]
            h = fafm["h"]
            w = fafm["w"]
        afm_line[:, :, 0] *= 128 / h
        afm_line[:, :, 1] *= 128 / w

        jun_c = []
        for line, score in zip(afm_line, afm_score):
            jun_c.append(list(line[0]) + [score])
            jun_c.append(list(line[1]) + [score])
        jun_c = np.array(jun_c)

        all_junc = np.vstack((all_junc, jun_c))
        all_jc_gt.append(junc_gt)
        all_junc_ids = np.hstack((all_junc_ids, np.array([i] * len(jun_c))))
    all_junc_ids = all_junc_ids.astype(np.int64)
    ap_jc = mAPJ(all_junc, all_jc_gt, DIST, all_junc_ids)
    print(f"  {ap_jc:.1f}")


def load_wf():
    pts = [defaultdict(int) for _ in range(500)]
    for thres in range(10):
        mats = sorted(glob.glob(f"{WF}/{thres}/*.mat"))
        for i, mat in enumerate(mats):
            img = cv2.imread(mat.replace(".mat", "_5.png"))
            juncs = loadmat(mat)["junctions"]
            if len(juncs) == 0:
                continue
            juncs[:, 0] *= 128 / img.shape[1]
            juncs[:, 1] *= 128 / img.shape[0]
            # juncs += 0.5
            for j in juncs:
                pts[i][tuple(j)] += 1
    pts = pts[: len(mats)]
    return [np.array([(k[1], k[0], v) for k, v in ipts.items()]) for ipts in pts]


def main():
    args = docopt(__doc__)
    gt_list = sorted(glob.glob(GT))
    im_list = sorted(glob.glob(IM))

    for path in args["<path>"]:
        print("Evaluating", path)
        lcnn_list = sorted(glob.glob(osp.join(path, "*.npz")))
        evaluate_lcnn(im_list, gt_list, lcnn_list)


if __name__ == "__main__":
    main()
