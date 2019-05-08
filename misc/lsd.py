#!/usr/bin/env python3

import os
import sys
import glob
import os.path as osp

import cv2
import numpy as np
import scipy.io
import matplotlib as mpl
import numpy.linalg as LA
import matplotlib.pyplot as plt

IM = "data/wireframe/valid-images/*.jpg"


if __name__ == "__main__":
    for i, iname in enumerate(sorted(glob.glob(IM))):
        img = cv2.imread(iname, 0)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        lsd_line, _, _, lsd_score = lsd.detect(img)
        lsd_line = lsd_line.reshape(-1, 2, 2)[:, :, ::-1]
        lsd_score = lsd_score.flatten()

        # plt.imshow(img)
        # for a, b in lsd_line:
        #     plt.plot([a[1], b[1]], [a[0], b[0]], linewidth=4)
        # plt.show()

        lsd_index = np.argsort(-lsd_score)
        np.savez_compressed(
            iname.replace(".jpg", "_LSD.npz"),
            lines=lsd_line[lsd_index],
            scores=lsd_score[lsd_index],
        )
