import math
import os.path as osp
import multiprocessing
from timeit import default_timer as timer

import numpy as np
import torch
import matplotlib.pyplot as plt


class benchmark(object):
    def __init__(self, msg, enable=True, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt
        self.enable = enable

    def __enter__(self):
        if self.enable:
            self.start = timer()
        return self

    def __exit__(self, *args):
        if self.enable:
            t = timer() - self.start
            print(("%s : " + self.fmt + " seconds") % (self.msg, t))
            self.time = t


def quiver(x, y, ax):
    ax.set_xlim(0, x.shape[1])
    ax.set_ylim(x.shape[0], 0)
    ax.quiver(
        x,
        y,
        units="xy",
        angles="xy",
        scale_units="xy",
        scale=1,
        minlength=0.01,
        width=0.1,
        color="b",
    )


def recursive_to(input, device):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict):
        for name in input:
            if isinstance(input[name], torch.Tensor):
                input[name] = input[name].to(device)
        return input
    if isinstance(input, list):
        for i, item in enumerate(input):
            input[i] = recursive_to(item, device)
        return input
    assert False


def np_softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def argsort2d(arr):
    return np.dstack(np.unravel_index(np.argsort(arr.ravel()), arr.shape))[0]


def __parallel_handle(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count(), progress_bar=lambda x: x):
    if nprocs == 0:
        nprocs = multiprocessing.cpu_count()
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [
        multiprocessing.Process(target=__parallel_handle, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    try:
        sent = [q_in.put((i, x)) for i, x in enumerate(X)]
        [q_in.put((None, None)) for _ in range(nprocs)]
        res = [q_out.get() for _ in progress_bar(range(len(sent)))]
        [p.join() for p in proc]
    except KeyboardInterrupt:
        q_in.close()
        q_out.close()
        raise
    return [x for i, x in sorted(res)]
