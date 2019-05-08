import os
import glob
import json
import math
import random

import numpy as np
import torch
import numpy.linalg as LA
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from lcnn.config import M


class WireframeDataset(Dataset):
    def __init__(
        self,
        rootdir,
        split,
    ):
        self.rootdir = rootdir
        filelist = glob.glob(f"{rootdir}/{split}/*_label.npz")
        filelist.sort()

        print(f"n{split}:", len(filelist))
        self.split = split
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        iname = self.filelist[idx][:-10].replace("_a0", "").replace("_a1", "") + ".png"
        image = io.imread(iname).astype(float)[:, :, :3]
        if "a1" in self.filelist[idx]:
            image = image[:, ::-1, :]
        image = (image - M.image.mean) / M.image.stddev
        image = np.rollaxis(image, 2).copy()

        with np.load(self.filelist[idx]) as npz:
            target = {
                name: torch.from_numpy(npz[name]).float()
                for name in ["jmap", "joff", "lmap"]
            }
            lpos = np.random.permutation(npz["lpos"])[: M.n_stc_posl]
            lneg = np.random.permutation(npz["lneg"])[: M.n_stc_negl]
            npos, nneg = len(lpos), len(lneg)
            lpre = np.concatenate([lpos, lneg], 0)
            for i in range(len(lpre)):
                if random.random() > 0.5:
                    lpre[i] = lpre[i, ::-1]
            ldir = lpre[:, 0, :2] - lpre[:, 1, :2]
            ldir /= np.clip(LA.norm(ldir, axis=1, keepdims=True), 1e-6, None)
            feat = [
                lpre[:, :, :2].reshape(-1, 4) / 128 * M.use_cood,
                ldir * M.use_slop,
                lpre[:, :, 2],
            ]
            feat = np.concatenate(feat, 1)
            meta = {
                "junc": torch.from_numpy(npz["junc"][:, :2]),
                "jtyp": torch.from_numpy(npz["junc"][:, 2]).byte(),
                "Lpos": self.adjacency_matrix(len(npz["junc"]), npz["Lpos"]),
                "Lneg": self.adjacency_matrix(len(npz["junc"]), npz["Lneg"]),
                "lpre": torch.from_numpy(lpre[:, :, :2]),
                "lpre_label": torch.cat([torch.ones(npos), torch.zeros(nneg)]),
                "lpre_feat": torch.from_numpy(feat),
            }

        return torch.from_numpy(image).float(), meta, target

    def adjacency_matrix(self, n, link):
        mat = torch.zeros(n + 1, n + 1, dtype=torch.uint8)
        link = torch.from_numpy(link)
        if len(link) > 0:
            mat[link[:, 0], link[:, 1]] = 1
            mat[link[:, 1], link[:, 0]] = 1
        return mat


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        [b[1] for b in batch],
        default_collate([b[2] for b in batch]),
    )
