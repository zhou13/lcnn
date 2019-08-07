#!/usr/bin/env python3

import sys
import glob
import os.path as osp

import cv2
import numpy as np
import scipy.io
import matplotlib as mpl
import numpy.linalg as LA
import matplotlib.pyplot as plt

try:
    sys.path.append(".")
    sys.path.append("..")
    import lcnn.utils
    import lcnn.metric
except Exception:
    raise

# Change the directory here
PRED = "logs/190418-201834-f8934c6-lr4d10/npz/000312000/*.npz"
PRED = "post/jmap_0008/*.npz"
GT = "data/wireframe/valid/*.npz"
# PRED = "logs/190506-001532-york/*.npz"
# GT = "data/york/valid/*.npz"
WF = "/data/lcnn/wirebase/result/wireframe/wireframe_1_rerun-baseline_0.5_0.5/*"
AFM = "/data/lcnn/wirebase/result/wireframe/afm/*.npz"


mpl.rcParams.update({"font.size": 16})
plt.rcParams["font.family"] = "Times New Roman"
del mpl.font_manager.weight_dict["roman"]
mpl.font_manager._rebuild()


def wireframe_score(T=10):
    gts = glob.glob(GT)
    gts.sort()
    dirs = glob.glob(WF)
    dirs.sort(key=lambda x: -float(osp.split(x)[-1]))

    precision, recall = [], []
    for threshold in dirs:
        print("Processing", threshold)
        mat_files = glob.glob(osp.join(threshold, "*.mat"))
        mat_files.sort()
        tp, fp, total_gt = 0, 0, 0
        for i, (gt_name, matf) in enumerate(zip(gts, mat_files)):
            line_pred = scipy.io.loadmat(matf)["lines"].reshape(-1, 2, 2)
            img = cv2.imread(matf.replace(".mat", ".jpg"))
            line_pred[:, :, 0] *= 128 / img.shape[1]
            line_pred[:, :, 1] *= 128 / img.shape[0]
            line_pred = line_pred[:, :, ::-1]

            with np.load(gt_name) as fgt:
                line_gt = fgt["lpos"][:, :, :2]
            tp_, fp_ = lcnn.metric.msTPFP(line_pred, line_gt, T)
            tp += tp_.sum()
            fp += fp_.sum()
            total_gt += len(line_gt)
        recall.append(tp / total_gt)
        precision.append(tp / (tp + fp))

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

    np.savez(
        "/data/lcnn/results/sAP/wireframe.npz",
        x=np.maximum(0.005, recall[:-1]),
        y=precision[:-1],
    )
    plt.plot(
        np.maximum(0.005, recall[:-1]),
        precision[:-1],
        label="Wireframe",
        linewidth=3,
        c="C1",
    )
    print("Huang sAP:", ap)


def line_score(threshold=10):
    preds = sorted(glob.glob(PRED))
    gts = sorted(glob.glob(GT))
    afm = sorted(glob.glob(AFM))

    lcnn_tp, lcnn_fp, lcnn_scores = [], [], []
    lsd_tp, lsd_fp, lsd_scores = [], [], []
    afm_tp, afm_fp, afm_scores = [], [], []
    n_gt = 0
    for pred_name, gt_name, afm_name in zip(preds, gts, afm):
        image = gt_name.replace("_label.npz", ".png")

        img = cv2.imread(image, 0)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        lsd_line, _, _, lsd_score = lsd.detect(img)
        lsd_line = lsd_line.reshape(-1, 2, 2)[:, :, ::-1]
        lsd_score = lsd_score.flatten()
        # print(lines.shape)
        # print(nfa.shape)

        with np.load(pred_name) as fpred:
            lcnn_line = fpred["lines"][:, :, :2]
            lcnn_score = fpred["score"]
        lcnn_line = lcnn_line[:, :, :2]
        with np.load(gt_name) as fgt:
            gt_line = fgt["lpos"][:, :, :2]

        with np.load(afm_name) as fafm:
            afm_line = fafm["lines"].reshape(-1, 2, 2)[:, :, ::-1]
            afm_score = -fafm["scores"]
            h = fafm["h"]
            w = fafm["w"]
        afm_line[:, :, 0] *= 128 / h
        afm_line[:, :, 1] *= 128 / w
        for i, ((a, b), s) in enumerate(zip(lcnn_line, lcnn_score)):
            if i > 0 and (lcnn_line[i] == lcnn_line[0]).all():
                lcnn_line = lcnn_line[:i]
                lcnn_score = lcnn_score[:i]
                break

        # plt.figure("LCNN")
        # for a, b in lcnn_line:
        #     plt.plot([a[1], b[1]], [a[0], b[0]], linewidth=4)
        # plt.figure("GT")
        # for a, b in gt_line:
        #     plt.plot([a[1], b[1]], [a[0], b[0]], linewidth=4)
        # plt.figure("LSD")
        # for a, b in lsd_line:
        #     plt.plot([a[1], b[1]], [a[0], b[0]], linewidth=4)
        # plt.figure("AFM")
        # for a, b in afm_line:
        #     plt.plot([a[1], b[1]], [a[0], b[0]], linewidth=4)
        # plt.show()

        tp, fp = lcnn.metric.msTPFP(lcnn_line, gt_line, threshold)
        lcnn_tp.append(tp)
        lcnn_fp.append(fp)
        lcnn_scores.append(lcnn_score)

        tp, fp = lcnn.metric.msTPFP(lsd_line, gt_line, threshold)
        lsd_tp.append(tp)
        lsd_fp.append(fp)
        lsd_scores.append(lsd_score)

        tp, fp = lcnn.metric.msTPFP(afm_line, gt_line, threshold)
        afm_tp.append(tp)
        afm_fp.append(fp)
        afm_scores.append(afm_score)

        n_gt += len(gt_line)

    lcnn_tp = np.concatenate(lcnn_tp)
    lcnn_fp = np.concatenate(lcnn_fp)
    lcnn_scores = np.concatenate(lcnn_scores)
    lcnn_index = np.argsort(-lcnn_scores)
    lcnn_tp = lcnn_tp[lcnn_index]
    lcnn_fp = lcnn_fp[lcnn_index]
    lcnn_tp = np.cumsum(lcnn_tp) / n_gt
    lcnn_fp = np.cumsum(lcnn_fp) / n_gt

    lsd_tp = np.concatenate(lsd_tp)
    lsd_fp = np.concatenate(lsd_fp)
    lsd_scores = np.concatenate(lsd_scores)
    lsd_index = np.argsort(-lsd_scores)
    lsd_tp = lsd_tp[lsd_index]
    lsd_fp = lsd_fp[lsd_index]
    lsd_tp = np.cumsum(lsd_tp) / n_gt
    lsd_fp = np.cumsum(lsd_fp) / n_gt

    afm_tp = np.concatenate(afm_tp)
    afm_fp = np.concatenate(afm_fp)
    afm_scores = np.concatenate(afm_scores)
    afm_index = np.argsort(-afm_scores)
    afm_tp = afm_tp[afm_index]
    afm_fp = afm_fp[afm_index]
    afm_tp = np.cumsum(afm_tp) / n_gt
    afm_fp = np.cumsum(afm_fp) / n_gt

    lcnn_re, lcnn_pr = lcnn_tp, lcnn_tp / (lcnn_tp + lcnn_fp)
    afm_re, afm_pr = afm_tp, afm_tp / (afm_tp + afm_fp)
    # lsd_re, lsd_pr = lsd_tp, lsd_tp / (lsd_tp + lsd_fp)

    T = 0.005
    plt.plot(afm_re[afm_re > T], afm_pr[afm_re > T], label="AFM", linewidth=3, c="C2")
    plt.plot(
        lcnn_re[lcnn_re > T], lcnn_pr[lcnn_re > T], label="L-CNN", linewidth=3, c="C3"
    )
    np.savez(
        "/data/lcnn/results/sAP/afm.npz", x=afm_re[afm_re > T], y=afm_pr[afm_re > T]
    )
    np.savez(
        "/data/lcnn/results/sAP/lcnn.npz", x=lcnn_re[lcnn_re > T], y=lcnn_pr[lcnn_re > T]
    )
    # plt.plot(lsd_re, lsd_pr, label="LSD", linewidth=2)

    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.yticks(np.arange(0, 1.0, step=0.1))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")

    f_scores = np.linspace(0.2, 0.8, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color="green", alpha=0.3)
        plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4)
    plt.title("PR Curve for sAP10")
    plt.savefig("sAP.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("sAP.svg", format="svg", bbox_inches="tight")
    plt.show()

    print(
        f"Processing {PRED}:\n"
        + f"    LSD sAP{threshold}: {lcnn.metric.ap(lsd_tp, lsd_fp)}\n"
        + f"    AFM sAP{threshold}: {lcnn.metric.ap(afm_tp, afm_fp)}\n"
        + f"    L-CNN sAP{threshold}: {lcnn.metric.ap(lcnn_tp, lcnn_fp)}"
    )


cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


if __name__ == "__main__":
    plt.tight_layout()
    wireframe_score()
    line_score()
