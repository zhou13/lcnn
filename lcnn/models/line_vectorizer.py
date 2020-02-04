import itertools
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lcnn.config import M

FEATURE_DIM = 8


class LineVectorizer(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        lambda_ = torch.linspace(0, 1, M.n_pts0)[:, None]
        self.register_buffer("lambda_", lambda_)
        self.do_static_sampling = M.n_stc_posl + M.n_stc_negl > 0

        self.fc1 = nn.Conv2d(256, M.dim_loi, 1)
        scale_factor = M.n_pts0 // M.n_pts1
        if M.use_conv:
            self.pooling = nn.Sequential(
                nn.MaxPool1d(scale_factor, scale_factor),
                Bottleneck1D(M.dim_loi, M.dim_loi),
            )
            self.fc2 = nn.Sequential(
                nn.ReLU(inplace=True), nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, 1)
            )
        else:
            self.pooling = nn.MaxPool1d(scale_factor, scale_factor)
            self.fc2 = nn.Sequential(
                nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, M.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(M.dim_fc, M.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(M.dim_fc, 1),
            )
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_dict):
        result = self.backbone(input_dict)
        h = result["preds"]
        x = self.fc1(result["feature"])
        n_batch, n_channel, row, col = x.shape

        xs, ys, fs, ps, idx, jcs = [], [], [], [], [0], []
        for i, meta in enumerate(input_dict["meta"]):
            p, label, feat, jc = self.sample_lines(
                meta, h["jmap"][i], h["joff"][i], input_dict["mode"]
            )
            # print("p.shape:", p.shape)
            ys.append(label)
            if input_dict["mode"] == "training" and self.do_static_sampling:
                p = torch.cat([p, meta["lpre"]])
                feat = torch.cat([feat, meta["lpre_feat"]])
                ys.append(meta["lpre_label"])
                del jc
            else:
                jcs.append(jc)
                ps.append(p)
            fs.append(feat)

            p = p[:, 0:1, :] * self.lambda_ + p[:, 1:2, :] * (1 - self.lambda_) - 0.5
            p = p.reshape(-1, 2)  # [N_LINE x N_POINT, 2_XY]
            px, py = p[:, 0].contiguous(), p[:, 1].contiguous()
            px0 = px.floor().clamp(min=0, max=127)
            py0 = py.floor().clamp(min=0, max=127)
            px1 = (px0 + 1).clamp(min=0, max=127)
            py1 = (py0 + 1).clamp(min=0, max=127)
            px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

            # xp: [N_LINE, N_CHANNEL, N_POINT]
            xp = (
                (
                    x[i, :, px0l, py0l] * (px1 - px) * (py1 - py)
                    + x[i, :, px1l, py0l] * (px - px0) * (py1 - py)
                    + x[i, :, px0l, py1l] * (px1 - px) * (py - py0)
                    + x[i, :, px1l, py1l] * (px - px0) * (py - py0)
                )
                .reshape(n_channel, -1, M.n_pts0)
                .permute(1, 0, 2)
            )
            xp = self.pooling(xp)
            xs.append(xp)
            idx.append(idx[-1] + xp.shape[0])

        x, y = torch.cat(xs), torch.cat(ys)
        f = torch.cat(fs)
        x = x.reshape(-1, M.n_pts1 * M.dim_loi)
        x = torch.cat([x, f], 1)
        x = self.fc2(x).flatten()

        if input_dict["mode"] != "training":
            p = torch.cat(ps)
            s = torch.sigmoid(x)
            b = s > 0.5
            lines = []
            score = []
            for i in range(n_batch):
                p0 = p[idx[i] : idx[i + 1]]
                s0 = s[idx[i] : idx[i + 1]]
                mask = b[idx[i] : idx[i + 1]]
                p0 = p0[mask]
                s0 = s0[mask]
                if len(p0) == 0:
                    lines.append(torch.zeros([1, M.n_out_line, 2, 2], device=p.device))
                    score.append(torch.zeros([1, M.n_out_line], device=p.device))
                else:
                    arg = torch.argsort(s0, descending=True)
                    p0, s0 = p0[arg], s0[arg]
                    lines.append(p0[None, torch.arange(M.n_out_line) % len(p0)])
                    score.append(s0[None, torch.arange(M.n_out_line) % len(s0)])
                for j in range(len(jcs[i])):
                    if len(jcs[i][j]) == 0:
                        jcs[i][j] = torch.zeros([M.n_out_junc, 2], device=p.device)
                    jcs[i][j] = jcs[i][j][
                        None, torch.arange(M.n_out_junc) % len(jcs[i][j])
                    ]
            result["preds"]["lines"] = torch.cat(lines)
            result["preds"]["score"] = torch.cat(score)
            result["preds"]["juncs"] = torch.cat([jcs[i][0] for i in range(n_batch)])
            if len(jcs[i]) > 1:
                result["preds"]["junts"] = torch.cat(
                    [jcs[i][1] for i in range(n_batch)]
                )

        if input_dict["mode"] != "testing":
            y = torch.cat(ys)
            loss = self.loss(x, y)
            lpos_mask, lneg_mask = y, 1 - y
            loss_lpos, loss_lneg = loss * lpos_mask, loss * lneg_mask

            def sum_batch(x):
                xs = [x[idx[i] : idx[i + 1]].sum()[None] for i in range(n_batch)]
                return torch.cat(xs)

            lpos = sum_batch(loss_lpos) / sum_batch(lpos_mask).clamp(min=1)
            lneg = sum_batch(loss_lneg) / sum_batch(lneg_mask).clamp(min=1)
            result["losses"][0]["lpos"] = lpos * M.loss_weight["lpos"]
            result["losses"][0]["lneg"] = lneg * M.loss_weight["lneg"]

        if input_dict["mode"] == "training":
            del result["preds"]

        return result

    def sample_lines(self, meta, jmap, joff, mode):
        with torch.no_grad():
            junc = meta["junc"]  # [N, 2]
            jtyp = meta["jtyp"]  # [N]
            Lpos = meta["Lpos"]
            Lneg = meta["Lneg"]

            n_type = jmap.shape[0]
            jmap = non_maximum_suppression(jmap).reshape(n_type, -1)
            joff = joff.reshape(n_type, 2, -1)
            max_K = M.n_dyn_junc // n_type
            N = len(junc)
            if mode != "training":
                K = min(int((jmap > M.eval_junc_thres).float().sum().item()), max_K)
            else:
                K = min(int(N * 2 + 2), max_K)
            if K < 2:
                K = 2
            device = jmap.device

            # index: [N_TYPE, K]
            score, index = torch.topk(jmap, k=K)
            y = (index / 128).float() + torch.gather(joff[:, 0], 1, index) + 0.5
            x = (index % 128).float() + torch.gather(joff[:, 1], 1, index) + 0.5

            # xy: [N_TYPE, K, 2]
            xy = torch.cat([y[..., None], x[..., None]], dim=-1)
            xy_ = xy[..., None, :]
            del x, y, index

            # dist: [N_TYPE, K, N]
            dist = torch.sum((xy_ - junc) ** 2, -1)
            cost, match = torch.min(dist, -1)

            # xy: [N_TYPE * K, 2]
            # match: [N_TYPE, K]
            for t in range(n_type):
                match[t, jtyp[match[t]] != t] = N
            match[cost > 1.5 * 1.5] = N
            match = match.flatten()

            _ = torch.arange(n_type * K, device=device)
            u, v = torch.meshgrid(_, _)
            u, v = u.flatten(), v.flatten()
            up, vp = match[u], match[v]
            label = Lpos[up, vp]

            if mode == "training":
                c = torch.zeros_like(label, dtype=torch.bool)

                # sample positive lines
                cdx = label.nonzero().flatten()
                if len(cdx) > M.n_dyn_posl:
                    # print("too many positive lines")
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_posl]
                    cdx = cdx[perm]
                c[cdx] = 1

                # sample negative lines
                cdx = Lneg[up, vp].nonzero().flatten()
                if len(cdx) > M.n_dyn_negl:
                    # print("too many negative lines")
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_negl]
                    cdx = cdx[perm]
                c[cdx] = 1

                # sample other (unmatched) lines
                cdx = torch.randint(len(c), (M.n_dyn_othr,), device=device)
                c[cdx] = 1
            else:
                c = (u < v).flatten()

            # sample lines
            u, v, label = u[c], v[c], label[c]
            xy = xy.reshape(n_type * K, 2)
            xyu, xyv = xy[u], xy[v]

            u2v = xyu - xyv
            u2v /= torch.sqrt((u2v ** 2).sum(-1, keepdim=True)).clamp(min=1e-6)
            feat = torch.cat(
                [
                    xyu / 128 * M.use_cood,
                    xyv / 128 * M.use_cood,
                    u2v * M.use_slop,
                    (u[:, None] > K).float(),
                    (v[:, None] > K).float(),
                ],
                1,
            )
            line = torch.cat([xyu[:, None], xyv[:, None]], 1)

            xy = xy.reshape(n_type, K, 2)
            jcs = [xy[i, score[i] > 0.03] for i in range(n_type)]
            return line, label.float(), feat, jcs


def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask


class Bottleneck1D(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Bottleneck1D, self).__init__()

        planes = outplanes // 2
        self.op = nn.Sequential(
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv1d(inplanes, planes, kernel_size=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, outplanes, kernel_size=1),
        )

    def forward(self, x):
        return x + self.op(x)
