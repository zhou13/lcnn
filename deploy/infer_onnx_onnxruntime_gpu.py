# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/26/2021 10:12 AM
# @File:infer_net1_onnx.py
import argparse
import os
import time
import skimage.io
import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from imutils import paths
import onnxruntime  #cuda10.2==onnxruntime-gpu 1.5.2

from deploy.creepageDistanceModel.models_creepage import NORMALIZATION_HEIGHT, NORMALIZATION_WIDTH, postprocess
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


    image_paths = list(paths.list_images(opt.predict_dir))

    for image_path in image_paths[:10]:
        im_o = skimage.io.imread(image_path)
        if im_o.ndim == 2:
            im_o = np.repeat(im_o[:, :, None], 3, 2)
        im_o = im_o[:, :, :3]

        # 第一次慢，从第二次开始快：应该是硬件从休眠状态warmup，比如cpu从低功耗低频状态提升到正常状态。
        # db适合用gpu，而angle和crnn正好相反、用CPU更快。
        image, pad, w0, h0 = get_image(image_path, opt.predict_type)
        junc = torch.zeros(1, 2).cuda()
        jtyp = torch.zeros(1, dtype=torch.uint8).cuda()
        Lpos = torch.zeros(2, 2, dtype=torch.uint8).cuda()

        start = time.time()
        nlines, nscores = ort_session.run(['lines', "score"], {'image': image.numpy(), })
        nlines[:, :, 1] = (nlines[:, :, 1] - pad[0]) * w0 / (NORMALIZATION_WIDTH - pad[0] * 2)  # x
        nlines[:, :, 0] = (nlines[:, :, 0] - pad[1]) * h0 / (NORMALIZATION_HEIGHT - pad[1] * 2)  # y
        if "tb" == opt.predict_type:
            nlines=nlines[:,:,[1,0]]
            nlines[:, :, 0] = w0 - nlines[:, :, 0]  # y
        print(time.time() - start)

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
        plt.imshow(im_o)
        plt.show()
        plt.close()
