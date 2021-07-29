# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/26/2021 10:12 AM
# @File:infer_net1_onnx.py
import argparse
import time

import copy
import skimage.io
import matplotlib as mpl
import matplotlib.pyplot as plt
import pprint
import torch
from imutils import paths
import numpy as np
from deploy.creepageDistanceModel.config import M, C
from deploy.creepageDistanceModel.models_creepage import NORMALIZATION_HEIGHT, NORMALIZATION_WIDTH, postprocess, hg, \
    MultitaskHead, MultitaskLearner, LineVectorizer

from deploy.torch2onnx import get_image

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

def c(x):
    return sm.to_rgba(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="获取杭州人工认为有缺陷大图")
    parser.add_argument('--config_file',
                        default=r"/home/zengxh/workspace/lcnn/deploy/creepageDistanceModel/wireframe.yaml",
                        help="没有分的文件夹")
    parser.add_argument('--checkpoint_path',
                        default=r"/home/zengxh/workspace/lcnn/logs/210726-144038-88f281a-baseline/checkpoint_best.pth",
                        help="没有分的文件夹")
    parser.add_argument('--predict_dir',
                        default=r"/home/zengxh/medias/data/ext/creepageDistance/20210714/smallimg/tb/org",
                        help="没有分的文件夹")
    parser.add_argument('--predict_type',
                        default=r"tb",
                        help="没有分的文件夹")
    opt = parser.parse_args()
    config_file = opt.config_file
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    checkpoint = torch.load(opt.checkpoint_path, map_location='cpu')

    # Load model
    model = hg(
        depth=M.depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
    )
    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
    model.eval()


    image_paths = list(paths.list_images(opt.predict_dir))
    for image_path in image_paths[:10]:
        im_o = skimage.io.imread(image_path)
        if im_o.ndim == 2:
            im_o = np.repeat(im_o[:, :, None], 3, 2)
        im_o = im_o[:, :, :3]

        image,pad,w0,h0 = get_image(image_path, opt.predict_type)
        junc = torch.zeros(1, 2).cuda()
        jtyp = torch.zeros(1, dtype=torch.uint8).cuda()
        Lpos = torch.zeros(2, 2, dtype=torch.uint8).cuda()

        start = time.time()
        lines, score = model(image.cuda(),junc,jtyp,Lpos)

        lines = lines[0].cpu().numpy() / (int(NORMALIZATION_HEIGHT / 4), int(NORMALIZATION_WIDTH / 4)) * (NORMALIZATION_HEIGHT,NORMALIZATION_WIDTH)
        scores = score[0].detach().cpu().numpy()
        for i in range(1, len(lines)):
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        # postprocess lines to remove overlapped lines
        diag = (NORMALIZATION_HEIGHT ** 2 + NORMALIZATION_WIDTH ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

        if len(nscores) < 2:
            continue
        elif len(nscores) == 2:
            pass
        else:
            middle_x = (np.max(nlines[:, :, 1]) - np.min(nlines[:, :, 1])) / 2 + np.min(nlines[:, :, 1])
            x_mean =np.mean(nlines[:, :, 1], axis=1)
            line1 = nlines[np.where(x_mean > middle_x)][np.argmax(nscores[np.where(x_mean > middle_x)])]
            line2 = nlines[np.where(x_mean < middle_x)][np.argmax(nscores[np.where(x_mean < middle_x)])]
            nlines=np.concatenate((line1[None, :, :], line2[None, :, :]), axis=0)

        nlines[:, :, 1] = (nlines[:, :, 1] - pad[0]) * w0 / (NORMALIZATION_WIDTH - pad[0] * 2)  # x
        nlines[:, :, 0] = (nlines[:, :, 0] - pad[1]) * h0 / (NORMALIZATION_HEIGHT - pad[1] * 2)  # y
        result_lines = copy.deepcopy(nlines)
        if "tb" == opt.predict_type:
            result_lines[:, :, 1] = nlines[:, :, 0]  # x
            result_lines[:, :, 0] = w0 - nlines[:, :, 1]  # y

        print(time.time()-start)
        # for i, t in enumerate([0.01, 0.95, 0.96, 0.97, 0.98, 0.99]):
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        for (a, b), s in zip(result_lines, nscores[:2]):
            # if s < t:
            #     continue
            plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=2, zorder=s)
            plt.scatter(a[1], a[0], **PLTOPTS)
            plt.scatter(b[1], b[0], **PLTOPTS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(im_o)
        plt.show()
        plt.close()