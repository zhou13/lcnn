import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from lcnn.utils import argsort2d

DX = [0, 0, 1, -1, 1, 1, -1, -1]
DY = [1, -1, 0, 0, 1, -1, 1, -1]


def ap(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])


def APJ(vert_pred, vert_gt, max_distance, im_ids):
    if len(vert_pred) == 0:
        return 0

    vert_pred = np.array(vert_pred)
    vert_gt = np.array(vert_gt)

    confidence = vert_pred[:, -1]
    idx = np.argsort(-confidence)
    vert_pred = vert_pred[idx, :]
    im_ids = im_ids[idx]
    n_gt = sum(len(gt) for gt in vert_gt)

    nd = len(im_ids)
    tp, fp = np.zeros(nd, dtype=np.float), np.zeros(nd, dtype=np.float)
    hit = [[False for _ in j] for j in vert_gt]

    for i in range(nd):
        gt_juns = vert_gt[im_ids[i]]
        pred_juns = vert_pred[i][:-1]
        if len(gt_juns) == 0:
            continue
        dists = np.linalg.norm((pred_juns[None, :] - gt_juns), axis=1)
        choice = np.argmin(dists)
        dist = np.min(dists)
        if dist < max_distance and not hit[im_ids[i]][choice]:
            tp[i] = 1
            hit[im_ids[i]][choice] = True
        else:
            fp[i] = 1

    tp = np.cumsum(tp) / n_gt
    fp = np.cumsum(fp) / n_gt
    return ap(tp, fp)


def nms_j(heatmap, delta=1):
    heatmap = heatmap.copy()
    disable = np.zeros_like(heatmap, dtype=np.bool)
    for x, y in argsort2d(heatmap):
        for dx, dy in zip(DX, DY):
            xp, yp = x + dx, y + dy
            if not (0 <= xp < heatmap.shape[0] and 0 <= yp < heatmap.shape[1]):
                continue
            if heatmap[x, y] >= heatmap[xp, yp]:
                disable[xp, yp] = True
    heatmap[disable] *= 0.6
    return heatmap


def mAPJ(pred, truth, distances, im_ids):
    return sum(APJ(pred, truth, d, im_ids) for d in distances) / len(distances) * 100


def post_jheatmap(heatmap, offset=None, delta=1):
    heatmap = nms_j(heatmap, delta=delta)
    # only select the best 1000 junctions for efficiency
    v0 = argsort2d(-heatmap)[:1000]
    confidence = -np.sort(-heatmap.ravel())[:1000]
    keep_id = np.where(confidence >= 1e-2)[0]
    if len(keep_id) == 0:
        return np.zeros((0, 3))

    confidence = confidence[keep_id]
    if offset is not None:
        v0 = np.array([v + offset[:, v[0], v[1]] for v in v0])
    v0 = v0[keep_id] + 0.5
    v0 = np.hstack((v0, confidence[:, np.newaxis]))
    return v0


def vectorized_wireframe_2d_metric(
    vert_pred, dpth_pred, edge_pred, vert_gt, dpth_gt, edge_gt, threshold
):
    # staging 1: matching
    nd = len(vert_pred)
    sorted_confidence = np.argsort(-vert_pred[:, -1])
    vert_pred = vert_pred[sorted_confidence, :-1]
    dpth_pred = dpth_pred[sorted_confidence]
    d = np.sqrt(
        np.sum(vert_pred ** 2, 1)[:, None]
        + np.sum(vert_gt ** 2, 1)[None, :]
        - 2 * vert_pred @ vert_gt.T
    )
    choice = np.argmin(d, 1)
    dist = np.min(d, 1)

    # staging 2: compute depth metric: SIL/L2
    loss_L1 = loss_L2 = 0
    hit = np.zeros_like(dpth_gt, np.bool)
    SIL = np.zeros(dpth_pred)
    for i in range(nd):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            loss_L1 += abs(dpth_gt[choice[i]] - dpth_pred[i])
            loss_L2 += (dpth_gt[choice[i]] - dpth_pred[i]) ** 2
            a = np.maximum(-dpth_pred[i], 1e-10)
            b = -dpth_gt[choice[i]]
            SIL[i] = np.log(a) - np.log(b)
        else:
            choice[i] = -1

    n = max(np.sum(hit), 1)
    loss_L1 /= n
    loss_L2 /= n
    loss_SIL = np.sum(SIL ** 2) / n - np.sum(SIL) ** 2 / (n * n)

    # staging 3: compute mAP for edge matching
    edgeset = set([frozenset(e) for e in edge_gt])
    tp = np.zeros(len(edge_pred), dtype=np.float)
    fp = np.zeros(len(edge_pred), dtype=np.float)
    for i, (v0, v1, score) in enumerate(sorted(edge_pred, key=-edge_pred[2])):
        length = LA.norm(vert_gt[v0] - vert_gt[v1], axis=1)
        if frozenset([choice[v0], choice[v1]]) in edgeset:
            tp[i] = length
        else:
            fp[i] = length
    total_length = LA.norm(
        vert_gt[edge_gt[:, 0]] - vert_gt[edge_gt[:, 1]], axis=1
    ).sum()
    return ap(tp / total_length, fp / total_length), (loss_SIL, loss_L1, loss_L2)


def vectorized_wireframe_3d_metric(
    vert_pred, dpth_pred, edge_pred, vert_gt, dpth_gt, edge_gt, threshold
):
    # staging 1: matching
    nd = len(vert_pred)
    sorted_confidence = np.argsort(-vert_pred[:, -1])
    vert_pred = np.hstack([vert_pred[:, :-1], dpth_pred[:, None]])[sorted_confidence]
    vert_gt = np.hstack([vert_gt[:, :-1], dpth_gt[:, None]])
    d = np.sqrt(
        np.sum(vert_pred ** 2, 1)[:, None]
        + np.sum(vert_gt ** 2, 1)[None, :]
        - 2 * vert_pred @ vert_gt.T
    )
    choice = np.argmin(d, 1)
    dist = np.min(d, 1)
    hit = np.zeros_like(dpth_gt, np.bool)
    for i in range(nd):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
        else:
            choice[i] = -1

    # staging 2: compute mAP for edge matching
    edgeset = set([frozenset(e) for e in edge_gt])
    tp = np.zeros(len(edge_pred), dtype=np.float)
    fp = np.zeros(len(edge_pred), dtype=np.float)
    for i, (v0, v1, score) in enumerate(sorted(edge_pred, key=-edge_pred[2])):
        length = LA.norm(vert_gt[v0] - vert_gt[v1], axis=1)
        if frozenset([choice[v0], choice[v1]]) in edgeset:
            tp[i] = length
        else:
            fp[i] = length
    total_length = LA.norm(
        vert_gt[edge_gt[:, 0]] - vert_gt[edge_gt[:, 1]], axis=1
    ).sum()

    return ap(tp / total_length, fp / total_length)


def msTPFP(line_pred, line_gt, threshold):
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), np.bool)
    tp = np.zeros(len(line_pred), np.float)
    fp = np.zeros(len(line_pred), np.float)
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def msAP(line_pred, line_gt, threshold):
    tp, fp = msTPFP(line_pred, line_gt, threshold)
    tp = np.cumsum(tp) / len(line_gt)
    fp = np.cumsum(fp) / len(line_gt)
    return ap(tp, fp)
