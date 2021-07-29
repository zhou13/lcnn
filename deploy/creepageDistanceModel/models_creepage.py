import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import M


# 用于归一化的宽度值
NORMALIZATION_WIDTH = 64
NORMALIZATION_HEIGHT = 512
# 像素最大值为255
PIXS_MAX_VALUE = 255.0
# 数据类型
TB_DATATYPE = "tb"
LR_DATATYPE = "lr"
# 准确率容错距离
ACC_PX_THRESH=16
# 随机种子
RANDOM_SEED = 1024

__all__ = ["HourglassNet", "hg"]


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, block, head, depth, num_stacks, num_blocks, num_classes):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * block.expansion
        # vpts = []
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, depth))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(head(ch, num_classes))
            # vpts.append(VptsHead(ch))
            # vpts.append(nn.Linear(ch, 9))
            # score.append(nn.Conv2d(ch, num_classes, kernel_size=1))
            # score[i].bias.data[0] += 4.6
            # score[i].bias.data[2] += 4.6
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        # self.vpts = nn.ModuleList(vpts)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out = []
        # out_vps = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            # pre_vpts = F.adaptive_avg_pool2d(x, (1, 1))
            # pre_vpts = pre_vpts.reshape(-1, 256)
            # vpts = self.vpts[i](x)
            out.append(score)
            # out_vps.append(vpts)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out[::-1], y  # , out_vps[::-1]


def hg(**kwargs):
    model = HourglassNet(
        Bottleneck2D,
        head=kwargs.get("head", lambda c_in, c_out: nn.Conv2D(c_in, c_out, 1)),
        depth=kwargs["depth"],
        num_stacks=kwargs["num_stacks"],
        num_blocks=kwargs["num_blocks"],
        num_classes=kwargs["num_classes"],
    )
    return model


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
        self.pooling = nn.MaxPool1d(scale_factor, scale_factor)
        self.fc2 = nn.Sequential(
            nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, M.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(M.dim_fc, M.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(M.dim_fc, 1),
        )

    def forward(self, image,junc,jtyp,Lpos):
        result = self.backbone(image)
        h = result["preds"]
        x = self.fc1(result["feature"])
        n_batch, n_channel, row, col = x.shape

        xs, ys, fs, ps, idx, = [], [], [], [], [0]
        i = 0
        p, label, feat = self.sample_lines(
            junc,jtyp,Lpos, h["jmap"][i], h["joff"][i]
        )
        # print("p.shape:", p.shape)
        ys.append(label)
        ps.append(p)
        fs.append(feat)

        p = p[:, 0:1, :] * self.lambda_ + p[:, 1:2, :] * (1 - self.lambda_) - 0.5
        p = p.reshape(-1, 2)  # [N_LINE x N_POINT, 2_XY]
        px, py = p[:, 1].contiguous(), p[:, 0].contiguous()
        px0 = px.floor().clamp(min=0, max=int(NORMALIZATION_WIDTH / 4)-1)
        py0 = py.floor().clamp(min=0, max=int(NORMALIZATION_HEIGHT / 4)-1)
        px1 = (px0 + 1).clamp(min=0, max=int(NORMALIZATION_WIDTH / 4)-1)
        py1 = (py0 + 1).clamp(min=0, max=int(NORMALIZATION_HEIGHT / 4)-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        # xp: [N_LINE, N_CHANNEL, N_POINT]
        xp = (
            (
                x[i, :, py0l, px0l] * (px1 - px) * (py1 - py)
                + x[i, :, py0l, px1l] * (px - px0) * (py1 - py)
                + x[i, :, py1l, px0l] * (px1 - px) * (py - py0)
                + x[i, :, py1l, px1l] * (px - px0) * (py - py0)
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
                v, arg = torch.sort(s0,descending=True)
                # arg = torch.argsort(s0, descending=True)
                p0, s0 = p0[arg], s0[arg]
                lines.append(p0[None, torch.arange(M.n_out_line) % len(p0)])
                score.append(s0[None, torch.arange(M.n_out_line) % len(s0)])
        return torch.cat(lines), torch.cat(score)

    def sample_lines(self, junc,jtyp,Lpos, jmap, joff):
        with torch.no_grad():
            n_type = jmap.shape[0]
            jmap = non_maximum_suppression(jmap).reshape(n_type, -1)
            # jmap = jmap.reshape(n_type, -1)
            joff = joff.reshape(n_type, 2, -1)
            max_K = M.n_dyn_junc // n_type
            N = len(junc)
            K = min(int((jmap > M.eval_junc_thres).float().sum().item()), max_K)
            if K < 2:
                K = 2
            device = jmap.device

            # index: [N_TYPE, K]
            score, index = torch.topk(jmap, k=K)
            y = (index / int(NORMALIZATION_WIDTH / 4)).float() + torch.gather(joff[:, 0], 1, index) + 0.5
            x = (index % int(NORMALIZATION_WIDTH / 4)).float() + torch.gather(joff[:, 1], 1, index) + 0.5

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

            c = u < v
            # sample lines
            u, v, label = u[c], v[c], label[c]
            xy = xy.reshape(n_type * K, 2)
            xyu, xyv = xy[u], xy[v]

            u2v = xyu - xyv
            u2v /= torch.sqrt((u2v ** 2).sum(-1, keepdim=True)).clamp(min=1e-6)
            feat = torch.cat(
                [
                    xyu / torch.tensor([int(NORMALIZATION_HEIGHT / 4),int(NORMALIZATION_WIDTH / 4)]).to(device) * M.use_cood,
                    xyv / torch.tensor([int(NORMALIZATION_HEIGHT / 4),int(NORMALIZATION_WIDTH / 4)]).to(device) * M.use_cood,
                    u2v * M.use_slop,
                    (u[:, None] > K).float(),
                    (v[:, None] > K).float(),
                ],
                1,
            )
            line = torch.cat([xyu[:, None], xyv[:, None]], 1)

            return line, label.float(), feat


def non_maximum_suppression(a):
    # output = F.max_pool2d(a, 3,stride=1)
    # ap = F.interpolate(output.unsqueeze(0), size=a.shape[1:], mode='bilinear', align_corners=True)
    # ap = ap.squeeze(0)
    # mask = (a == ap).float().clamp(min=0.0)
    # return a * mask

    # au = a.unsqueeze(0)
    # output, indices = F.max_pool2d(au, 3, stride=1, return_indices=True)
    # ap = F.max_unpool2d(output, indices, 3, stride=1,output_size=au.shape)
    # ap = ap.squeeze(0)
    # mask = (a == ap).float().clamp(min=0.0)
    # return a * mask
    # 等价于下面的
    ap = F.max_pool2d(a.unsqueeze(0), 3, stride=1, padding=1)
    ap = ap.squeeze(0)
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


class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(M.head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(M.head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class MultitaskLearner(nn.Module):
    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone
        head_size = M.head_size
        self.num_class = sum(sum(head_size, []))
        self.head_off = np.cumsum([sum(h) for h in head_size])

    def forward(self, image):
        outputs, feature = self.backbone(image)
        result = {"feature": feature}
        batch, channel, row, col = outputs[0].shape

        n_jtyp = 1 # batch_size

        offset = self.head_off
        output=outputs[0]
        output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
        jmap = output[0 : offset[0]].reshape(n_jtyp, 2, batch, row, col)
        joff = output[offset[1] : offset[2]].reshape(n_jtyp, 2, batch, row, col)
        result["preds"] = {
            "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
            "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
        }
        return result

def pline(x1, y1, x2, y2, x, y):
    px = x2 - x1
    py = y2 - y1
    dd = px * px + py * py
    u = ((x - x1) * px + (y - y1) * py) / max(1e-9, float(dd))
    dx = x1 + u * px - x
    dy = y1 + u * py - y
    return dx * dx + dy * dy


def plambda(x1, y1, x2, y2, x, y):
    px = x2 - x1
    py = y2 - y1
    dd = px * px + py * py
    return ((x - x1) * px + (y - y1) * py) / max(1e-9, float(dd))

def postprocess(lines, scores, threshold=0.01, tol=1e9, do_clip=False):
    nlines, nscores = [], []
    for (p, q), score in zip(lines, scores):
        start, end = 0, 1
        for a, b in nlines:
            if (
                min(
                    max(pline(*p, *q, *a), pline(*p, *q, *b)),
                    max(pline(*a, *b, *p), pline(*a, *b, *q)),
                )
                > threshold ** 2
            ):
                continue
            lambda_a = plambda(*p, *q, *a)
            lambda_b = plambda(*p, *q, *b)
            if lambda_a > lambda_b:
                lambda_a, lambda_b = lambda_b, lambda_a
            lambda_a -= tol
            lambda_b += tol

            # case 1: skip (if not do_clip)
            if start < lambda_a and lambda_b < end:
                continue

            # not intersect
            if lambda_b < start or lambda_a > end:
                continue

            # cover
            if lambda_a <= start and end <= lambda_b:
                start = 10
                break

            # case 2 & 3:
            if lambda_a <= start and start <= lambda_b:
                start = lambda_b
            if lambda_a <= end and end <= lambda_b:
                end = lambda_a

            if start >= end:
                break

        if start >= end:
            continue
        nlines.append(np.array([p + (q - p) * start, p + (q - p) * end]))
        nscores.append(score)
    return np.array(nlines), np.array(nscores)