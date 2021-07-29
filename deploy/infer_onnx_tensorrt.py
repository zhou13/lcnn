# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/26/2021 10:12 AM
# @File:infer_net1_onnx.py
import argparse
import onnx
import onnx_tensorrt.backend as backend
import torch
from imutils import paths

from deploy.torch2onnx import get_image

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

    model = onnx.load(opt.onnx_path)
    engine = backend.prepare(model, device='CUDA:0')

    image_paths = list(paths.list_images(opt.predict_dir))
    image_path = image_paths[0]
    image = get_image(image_path, opt.predict_type).cuda()
    junc = torch.zeros(1, 2).cuda()
    jtyp = torch.zeros(1, dtype=torch.uint8).cuda()
    Lpos = torch.zeros(2, 2, dtype=torch.uint8).cuda()

    ret = engine.run(image)
    print(ret)