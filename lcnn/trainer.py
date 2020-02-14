import atexit
import os
import os.path as osp
import shutil
import signal
import subprocess
import threading
import time
from timeit import default_timer as timer

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from tensorboardX import SummaryWriter

from lcnn.config import C, M
from lcnn.utils import recursive_to


class Trainer(object):
    def __init__(self, device, model, optimizer, train_loader, val_loader, out):
        self.device = device

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = C.model.batch_size

        self.validation_interval = C.io.validation_interval

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.run_tensorboard()
        time.sleep(1)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = C.optim.max_epoch
        self.lr_decay_epoch = C.optim.lr_decay_epoch
        self.num_stacks = C.model.num_stacks
        self.mean_loss = self.best_mean_loss = 1e1000

        self.loss_labels = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)

    def run_tensorboard(self):
        board_out = osp.join(self.out, "tensorboard")
        if not osp.exists(board_out):
            os.makedirs(board_out)
        self.writer = SummaryWriter(board_out)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        p = subprocess.Popen(
            ["tensorboard", f"--logdir={board_out}", f"--port={C.io.tensorboard_port}"]
        )

        def killme():
            os.kill(p.pid, signal.SIGTERM)

        atexit.register(killme)

    def _loss(self, result):
        losses = result["losses"]
        # Don't move loss label to other place.
        # If I want to change the loss, I just need to change this function.
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + list(losses[0].keys())
            self.metrics = np.zeros([self.num_stacks, len(self.loss_labels)])
            print()
            print(
                "| ".join(
                    ["progress "]
                    + list(map("{:7}".format, self.loss_labels))
                    + ["speed"]
                )
            )
            with open(f"{self.out}/loss.csv", "a") as fout:
                print(",".join(["progress"] + self.loss_labels), file=fout)

        total_loss = 0
        for i in range(self.num_stacks):
            for j, name in enumerate(self.loss_labels):
                if name == "sum":
                    continue
                if name not in losses[i]:
                    assert i != 0
                    continue
                loss = losses[i][name].mean()
                self.metrics[i, 0] += loss.item()
                self.metrics[i, j] += loss.item()
                total_loss += loss
        return total_loss

    def validate(self):
        tprint("Running validation...", " " * 75)
        training = self.model.training
        self.model.eval()

        viz = osp.join(self.out, "viz", f"{self.iteration * M.batch_size_eval:09d}")
        npz = osp.join(self.out, "npz", f"{self.iteration * M.batch_size_eval:09d}")
        osp.exists(viz) or os.makedirs(viz)
        osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        with torch.no_grad():
            for batch_idx, (image, meta, target) in enumerate(self.val_loader):
                input_dict = {
                    "image": recursive_to(image, self.device),
                    "meta": recursive_to(meta, self.device),
                    "target": recursive_to(target, self.device),
                    "mode": "validation",
                }
                result = self.model(input_dict)

                total_loss += self._loss(result)

                H = result["preds"]
                for i in range(H["jmap"].shape[0]):
                    index = batch_idx * M.batch_size_eval + i
                    np.savez(
                        f"{npz}/{index:06}.npz",
                        **{k: v[i].cpu().numpy() for k, v in H.items()},
                    )
                    if index >= 20:
                        continue
                    self._plot_samples(i, index, H, meta, target, f"{viz}/{index:06}")

        self._write_metrics(len(self.val_loader), total_loss, "validation", True)
        self.mean_loss = total_loss / len(self.val_loader)

        torch.save(
            {
                "iteration": self.iteration,
                "arch": self.model.__class__.__name__,
                "optim_state_dict": self.optim.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "best_mean_loss": self.best_mean_loss,
            },
            osp.join(self.out, "checkpoint_latest.pth"),
        )
        shutil.copy(
            osp.join(self.out, "checkpoint_latest.pth"),
            osp.join(npz, "checkpoint.pth"),
        )
        if self.mean_loss < self.best_mean_loss:
            self.best_mean_loss = self.mean_loss
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth"),
                osp.join(self.out, "checkpoint_best.pth"),
            )

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        time = timer()
        for batch_idx, (image, meta, target) in enumerate(self.train_loader):

            self.optim.zero_grad()
            self.metrics[...] = 0

            input_dict = {
                "image": recursive_to(image, self.device),
                "meta": recursive_to(meta, self.device),
                "target": recursive_to(target, self.device),
                "mode": "training",
            }
            result = self.model(input_dict)

            loss = self._loss(result)
            if np.isnan(loss.item()):
                raise ValueError("loss is nan while training")
            loss.backward()
            self.optim.step()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics = self.avg_metrics * 0.9 + self.metrics * 0.1
            self.iteration += 1
            self._write_metrics(1, loss.item(), "training", do_print=False)

            if self.iteration % 4 == 0:
                tprint(
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, self.avg_metrics[0]))
                    + f"| {4 * self.batch_size / (timer() - time):04.1f} "
                )
                time = timer()
            num_images = self.batch_size * self.iteration
            if num_images % self.validation_interval == 0 or num_images == 600:
                self.validate()
                time = timer()

    def _write_metrics(self, size, total_loss, prefix, do_print=False):
        for i, metrics in enumerate(self.metrics):
            for label, metric in zip(self.loss_labels, metrics):
                self.writer.add_scalar(
                    f"{prefix}/{i}/{label}", metric / size, self.iteration
                )
            if i == 0 and do_print:
                csv_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size:07},"
                    + ",".join(map("{:.11f}".format, metrics / size))
                )
                prt_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, metrics / size))
                )
                with open(f"{self.out}/loss.csv", "a") as fout:
                    print(csv_str, file=fout)
                pprint(prt_str, " " * 7)
        self.writer.add_scalar(
            f"{prefix}/total_loss", total_loss / size, self.iteration
        )
        return total_loss

    def _plot_samples(self, i, index, result, meta, target, prefix):
        fn = self.val_loader.dataset.filelist[index][:-10].replace("_a0", "") + ".png"
        img = io.imread(fn)
        imshow(img), plt.savefig(f"{prefix}_img.jpg"), plt.close()

        mask_result = result["jmap"][i].cpu().numpy()
        mask_target = target["jmap"][i].cpu().numpy()
        for ch, (ia, ib) in enumerate(zip(mask_target, mask_result)):
            imshow(ia), plt.savefig(f"{prefix}_mask_{ch}a.jpg"), plt.close()
            imshow(ib), plt.savefig(f"{prefix}_mask_{ch}b.jpg"), plt.close()

        line_result = result["lmap"][i].cpu().numpy()
        line_target = target["lmap"][i].cpu().numpy()
        imshow(line_target), plt.savefig(f"{prefix}_line_a.jpg"), plt.close()
        imshow(line_result), plt.savefig(f"{prefix}_line_b.jpg"), plt.close()

        def draw_vecl(lines, sline, juncs, junts, fn):
            imshow(img)
            if len(lines) > 0 and not (lines[0] == 0).all():
                for i, ((a, b), s) in enumerate(zip(lines, sline)):
                    if i > 0 and (lines[i] == lines[0]).all():
                        break
                    plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=4)
            if not (juncs[0] == 0).all():
                for i, j in enumerate(juncs):
                    if i > 0 and (i == juncs[0]).all():
                        break
                    plt.scatter(j[1], j[0], c="red", s=64, zorder=100)
            if junts is not None and len(junts) > 0 and not (junts[0] == 0).all():
                for i, j in enumerate(junts):
                    if i > 0 and (i == junts[0]).all():
                        break
                    plt.scatter(j[1], j[0], c="blue", s=64, zorder=100)
            plt.savefig(fn), plt.close()

        junc = meta[i]["junc"].cpu().numpy() * 4
        jtyp = meta[i]["jtyp"].cpu().numpy()
        juncs = junc[jtyp == 0]
        junts = junc[jtyp == 1]
        rjuncs = result["juncs"][i].cpu().numpy() * 4
        rjunts = None
        if "junts" in result:
            rjunts = result["junts"][i].cpu().numpy() * 4

        lpre = meta[i]["lpre"].cpu().numpy() * 4
        vecl_target = meta[i]["lpre_label"].cpu().numpy()
        vecl_result = result["lines"][i].cpu().numpy() * 4
        score = result["score"][i].cpu().numpy()
        lpre = lpre[vecl_target == 1]

        draw_vecl(lpre, np.ones(lpre.shape[0]), juncs, junts, f"{prefix}_vecl_a.jpg")
        draw_vecl(vecl_result, score, rjuncs, rjunts, f"{prefix}_vecl_b.jpg")

    def train(self):
        plt.rcParams["figure.figsize"] = (24, 24)
        # if self.iteration == 0:
        #     self.validate()
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size
        for self.epoch in range(start_epoch, self.max_epoch):
            if self.epoch == self.lr_decay_epoch:
                self.optim.param_groups[0]["lr"] /= 10
            self.train_epoch()


cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def imshow(im):
    plt.close()
    plt.tight_layout()
    plt.imshow(im)
    plt.colorbar(sm, fraction=0.046)
    plt.xlim([0, im.shape[0]])
    plt.ylim([im.shape[0], 0])


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


def _launch_tensorboard(board_out, port, out):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = subprocess.Popen(["tensorboard", f"--logdir={board_out}", f"--port={port}"])

    def kill():
        os.kill(p.pid, signal.SIGTERM)

    atexit.register(kill)
