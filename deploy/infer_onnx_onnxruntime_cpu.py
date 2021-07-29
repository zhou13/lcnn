# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/26/2021 10:12 AM
# @File:infer_net1_onnx.py
import argparse
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from imutils import paths
import onnxruntime  # onnxruntime 1.8.1

from deploy.creepageDistanceModel.models_creepage import NORMALIZATION_HEIGHT, NORMALIZATION_WIDTH, postprocess

print(onnxruntime.get_device())

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
    parser.add_argument('--devices',
                        default=r"0",
                        help="没有分的文件夹")
    parser.add_argument('--onnx_path',
                        default=r"/home/zengxh/workspace/lcnn/logs/210726-144038-88f281a-baseline/checkpoint_best.onnx",
                        help="没有分的文件夹")
    parser.add_argument('--predict_dir',
                        default=r"/home/zengxh/medias/data/ext/creepageDistance/20210714/smallimg/tb/org",
                        help="没有分的文件夹")
    parser.add_argument('--predict_type',
                        default=r"tb",
                        help="没有分的文件夹")
    opt = parser.parse_args()

    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    ort_session = onnxruntime.InferenceSession(opt.onnx_path,options)
    # ort_session.set_providers([ort_session.get_providers()[1]])  # 强制指定用CPU识别

    image_paths = list(paths.list_images(opt.predict_dir))
    for image_path in image_paths[:10]:
        im,image = get_image(image_path, opt.predict_type)
        junc = torch.zeros(1, 2).cuda()
        jtyp = torch.zeros(1, dtype=torch.uint8).cuda()
        Lpos = torch.zeros(2, 2, dtype=torch.uint8).cuda()

        start = time.time()
        lines, score = ort_session.run(['lines',"score"], {'image': image.numpy(),})
        print(time.time() - start)

        lines = lines[0] / (int(NORMALIZATION_HEIGHT / 4), int(NORMALIZATION_WIDTH / 4)) * im.shape[:2]
        scores = score[0]
        for i in range(1, len(lines)):
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        # postprocess lines to remove overlapped lines
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)
        # for i, t in enumerate([0.01, 0.95, 0.96, 0.97, 0.98, 0.99]):
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        for (a, b), s in zip(nlines, nscores):
            # if s < t:
            #     continue
            plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=2, zorder=s)
            plt.scatter(a[1], a[0], **PLTOPTS)
            plt.scatter(b[1], b[0], **PLTOPTS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(im)
        plt.show()
        plt.close()
