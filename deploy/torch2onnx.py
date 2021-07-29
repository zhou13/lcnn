# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/13/2021 1:06 PM
# @File:预测图片并且生成json标准文件
import argparse
import copy
import cv2
import os
import pprint
import random
import numpy as np
import onnx
import skimage
import torch
from imutils import paths
import torch.nn as nn

from dataset.constants import NORMALIZATION_HEIGHT, NORMALIZATION_WIDTH, TB_DATATYPE
from deploy.creepageDistanceModel.config import M, C
from deploy.creepageDistanceModel.models_creepage import MultitaskHead, MultitaskLearner, LineVectorizer, hg
from python_developer_tools.cv.utils.torch_utils import recursive_to, init_seeds, init_cudnn
from python_developer_tools.cv.datasets.datasets_utils import letterbox

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_image(image_path, datatype):
    img = cv2.imread(image_path)
    # origin_img = copy.deepcopy(img)
    if "tb" == datatype:
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)
    h0, w0 = img.shape[:2]  # orig hw
    im, ratio, pad = letterbox(img, [NORMALIZATION_HEIGHT, NORMALIZATION_WIDTH], auto=False, scaleFill=True)
    cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, 2)
    im = im[:, :, :3]
    im_resized = skimage.transform.resize(im, (NORMALIZATION_HEIGHT, NORMALIZATION_WIDTH)) * 255
    image = (im_resized - [109.730, 103.832, 98.681]) / [22.275, 22.124, 23.229]
    image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
    return image, pad, w0, h0


class mymodel(nn.Module):
    def __init__(self, ckpt_path):
        super(mymodel, self).__init__()
        checkpoint = torch.load(ckpt_path, map_location='cpu')

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

        self.model = model

        self.lines_tensor = torch.tensor([NORMALIZATION_HEIGHT / 4, NORMALIZATION_WIDTH / 4]).cuda()
        self.le9_tensor = torch.tensor(1e-9).cuda()
        self.zero_tensor = torch.tensor(0).cuda()
        self.one_tensor  = torch.tensor(1).cuda()
        self.ten_tensor  = torch.tensor(10).cuda()

    def pline(self, x1, y1, x2, y2, x, y):
        px = x2 - x1
        py = y2 - y1
        dd = px * px + py * py
        u = ((x - x1) * px + (y - y1) * py) / torch.max(self.le9_tensor,dd)
        dx = x1 + u * px - x
        dy = y1 + u * py - y
        return dx * dx + dy * dy

    def plambda(self, x1, y1, x2, y2, x, y):
        px = x2 - x1
        py = y2 - y1
        dd = px * px + py * py
        return ((x - x1) * px + (y - y1) * py) / torch.max(self.le9_tensor,dd)

    def postprocess(self, lines, scores, threshold=0.01, tol=1e9, do_clip=False):
        nlines, nscores = [], []
        for (p, q), score in zip(lines, scores):
            start, end = 0, 1
            for a, b in nlines:
                if (
                        torch.min(
                            torch.max(self.pline(*p, *q, *a), self.pline(*p, *q, *b)),
                            torch.max(self.pline(*a, *b, *p), self.pline(*a, *b, *q)),
                        )
                        > threshold ** 2
                ):
                    continue
                lambda_a = self.plambda(*p, *q, *a)
                lambda_b = self.plambda(*p, *q, *b)
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

            t_cat = torch.cat(((p + (q - p) * start).view(1, 2), (p + (q - p) * end).view(1, 2)), 0)
            nlines.append(t_cat)
            nscores.append(score)
        nlines = torch.cat([nline.unsqueeze(0) for nline in nlines])
        nscores = torch.tensor(nscores)
        return nlines, nscores


    def forward(self, image, junc, jtyp, Lpos):
        lines, score = self.model(image, junc, jtyp, Lpos)
        lines = torch.div(lines[0], torch.tensor([NORMALIZATION_HEIGHT / 4, NORMALIZATION_WIDTH / 4]).cuda())
        lines = torch.mul(lines, torch.tensor([NORMALIZATION_HEIGHT, NORMALIZATION_WIDTH]).cuda())
        scores = score[0]
        len_lines = lines.shape[0]
        for i in range(1, len_lines):
            if torch.equal(lines[i], lines[0]):
                lines = lines[:i]
                scores = scores[:i]
                break

        # postprocess lines to remove overlapped lines
        diag = (NORMALIZATION_HEIGHT ** 2 + NORMALIZATION_WIDTH ** 2) ** 0.5
        nlines, nscores = self.postprocess(lines, scores, diag * 0.01, 0, False)

        if len(nscores) > 2:
            xnlines = nlines[:, :, 1]

            # nonzeroindex = torch.nonzero(xnlines)
            nlines_min = torch.min( xnlines ) #torch.min( xnlines[nonzeroindex] )
            middle_x = (torch.max(xnlines) - nlines_min) / 2 + nlines_min
            x_mean = torch.mean(xnlines, axis=1)

            xmeangt = x_mean > middle_x
            xmeanlt = x_mean < middle_x
            indexline1 = torch.argmax(nscores[xmeangt])
            indexline2 = torch.argmax(nscores[xmeanlt])
            line1 = nlines[xmeangt][indexline1]
            line2 = nlines[xmeanlt][indexline2]
            nlines = torch.cat((line1.unsqueeze(0), line2.unsqueeze(0)), 0)
            return nlines, torch.tensor([nscores[xmeangt][indexline1],
                                         nscores[xmeanlt][indexline2]])
        return nlines, nscores




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="获取杭州人工认为有缺陷大图")
    parser.add_argument('--config_file',
                        default=r"/home/zengxh/workspace/lcnn/deploy/creepageDistanceModel/wireframe.yaml",
                        help="没有分的文件夹")
    parser.add_argument('--checkpoint_path',
                        default=r"/home/zengxh/workspace/lcnn/logs/210726-144038-88f281a-baseline/checkpoint_best.pth",
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
    config_file = opt.config_file
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    init_seeds(0)

    device_name = "cuda"
    init_cudnn()
    print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    device = torch.device(device_name)

    mymodel = mymodel(opt.checkpoint_path)

    image_paths = list(paths.list_images(opt.predict_dir))
    image_path = image_paths[0]
    image, pad, w0, h0 = get_image(image_path, opt.predict_type)
    junc = torch.zeros(1, 2).cuda()
    jtyp = torch.zeros(1, dtype=torch.uint8).cuda()
    Lpos = torch.zeros(2, 2, dtype=torch.uint8).cuda()
    torch.onnx.export(mymodel, (image.cuda(), junc, jtyp, Lpos), opt.onnx_path,
                      opset_version=11,
                      verbose=True,
                      export_params=True,  # 是否导出params
                      do_constant_folding=True,  # 是否进行常量折叠进行优化
                      # dynamic_axes= {
                      #                'input': {0: 'image',1: 'junc',2: 'jtyp',3: 'Lpos'},
                      #                'output': {0: 'lines',1: 'score'}
                      #                },
                      input_names=['image', "junc", "jtyp", "Lpos"],
                      output_names=['lines', "score"], )
    print("导出完成")
    # 检查导出的model
    onnxmodel = onnx.load(opt.onnx_path)
    onnx.checker.check_model(onnxmodel)
    print("检查完成")
