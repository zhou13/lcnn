import argparse
import os
import shutil
import random

import cv2
import yaml
from imutils import paths
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset.constants import NORMALIZATION_WIDTH, NORMALIZATION_HEIGHT, RANDOM_SEED
from python_developer_tools.cv.datasets.datasets_utils import resize_image, letterbox
from python_developer_tools.files.common import mkdir, get_filename_suf_pix
from python_developer_tools.files.json_utils import read_json_file, save_json_file

def createDatasets(datasets, dirname):
    dataDir = os.path.join(data_dict_tmp[dirname])
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
    for datapath in datasets:
        shutil.copy(datapath.replace(".jpg",".json"), dataDir)
        shutil.copy(datapath, dataDir)

def get_origin_image_points(imagePath):
    img = cv2.imread(imagePath)
    jsonfile = imagePath.replace(".jpg", ".json")
    json_cont = read_json_file(jsonfile)
    labels_tmp = [0, 0, 0, 0, 0, 0, 0, 0]
    for shapes in json_cont["shapes"]:
        label = shapes["label"]
        points = shapes["points"]
        if json_cont["imageHeight"] > json_cont["imageWidth"]:
            if label == "min":
                if points[0][1] < points[1][1]:
                    labels_tmp[4] = points[0][0]
                    labels_tmp[5] = points[0][1]
                    labels_tmp[6] = points[1][0]
                    labels_tmp[7] = points[1][1]
                else:
                    labels_tmp[4] = points[1][0]
                    labels_tmp[5] = points[1][1]
                    labels_tmp[6] = points[0][0]
                    labels_tmp[7] = points[0][1]
            if label == "max":
                if points[0][1] < points[1][1]:
                    labels_tmp[0] = points[0][0]
                    labels_tmp[1] = points[0][1]
                    labels_tmp[2] = points[1][0]
                    labels_tmp[3] = points[1][1]
                else:
                    labels_tmp[0] = points[1][0]
                    labels_tmp[1] = points[1][1]
                    labels_tmp[2] = points[0][0]
                    labels_tmp[3] = points[0][1]
        else:
            if label == "min":
                if points[0][0] < points[1][0]:
                    labels_tmp[4] = points[0][0]
                    labels_tmp[5] = points[0][1]
                    labels_tmp[6] = points[1][0]
                    labels_tmp[7] = points[1][1]
                else:
                    labels_tmp[4] = points[1][0]
                    labels_tmp[5] = points[1][1]
                    labels_tmp[6] = points[0][0]
                    labels_tmp[7] = points[0][1]
            if label == "max":
                if points[0][0] < points[1][0]:
                    labels_tmp[0] = points[0][0]
                    labels_tmp[1] = points[0][1]
                    labels_tmp[2] = points[1][0]
                    labels_tmp[3] = points[1][1]
                else:
                    labels_tmp[0] = points[1][0]
                    labels_tmp[1] = points[1][1]
                    labels_tmp[2] = points[0][0]
                    labels_tmp[3] = points[0][1]
    return img,labels_tmp

def label_transpose_1(label_o,w0,h0):
    # tb 顺时针旋转90°
    new_label = [0, 0, 0, 0, 0, 0, 0, 0]
    new_label[0] = w0-label_o[5]
    new_label[1] = label_o[4]
    new_label[2] = w0-label_o[7]
    new_label[3] = label_o[6]
    new_label[4] = w0-label_o[1]
    new_label[5] = label_o[0]
    new_label[6] = w0-label_o[3]
    new_label[7] = label_o[2]
    return new_label

def labels_convert_train(label,w0,h0,w1,h1,w2,h2,padw,padh):
    new_label = [0,0,0,0,0,0,0,0]
    for i,_label in enumerate(label):
        if i in [0,2,4,6]:
            new_label[i] = ((_label * w1 / w0)  * w2 ) / w1   + padw
        else:
            new_label[i] = ((_label * h1 / h0)  * h2 ) / h1   + padh
    # label = [i / w0 for i in label]
    # label = [i * w1 / w0 for i in label]
    # label = [(i * w2+ padw) / w1 for i in label]
    # label = [i / NORMALIZATION_WIDTH for i in label]
    return new_label

def get_dict_json(imagePath):
    filename, filedir, filesuffix, filenamestem = get_filename_suf_pix(imagePath)
    img, labels_tmp = get_origin_image_points(imagePath)
    if key == "tb":
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)
        h0, w0 = img.shape[:2]  # orig hw
        labels_tmp = label_transpose_1(labels_tmp, w0, h0)
        # _ = cv2.line(img, (int(labels_tmp[0]), int(labels_tmp[1])), (int(labels_tmp[2]), int(labels_tmp[3])),
        #              (0, 255, 0), thickness=2)
        # _ = cv2.line(img, (int(labels_tmp[4]), int(labels_tmp[5])), (int(labels_tmp[6]), int(labels_tmp[7])),
        #              (0, 0, 255), thickness=2)
        # cv2.imwrite("sdf.jpg", img)
    h0, w0 = img.shape[:2]

    img = resize_image(img, [NORMALIZATION_HEIGHT, NORMALIZATION_WIDTH])
    h1, w1, _ = img.shape
    img, ratio, pad = letterbox(img, [NORMALIZATION_HEIGHT, NORMALIZATION_WIDTH], auto=False, scaleup=True)
    #letterbox(img, img_size, auto=False,scaleFill=True)  # 会填充边缘 letterbox(img, self.opt.img_size, auto=False, scaleup=False)
    labels_tmp = labels_convert_train(labels_tmp, w0, h0, w1, h1, ratio[0] * w1, ratio[1] * h1, pad[0], pad[1])

    h2, w2 = img.shape[:2]
    dict_json = {"filename": filename,
                 "lines": [[labels_tmp[0], labels_tmp[1], labels_tmp[2], labels_tmp[3]],
                           [labels_tmp[4], labels_tmp[5], labels_tmp[6], labels_tmp[7]]],
                 "height": h2, "width": w2}
    cv2.imwrite(os.path.join(images_dir, filename), img)
    return dict_json

if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    parser = argparse.ArgumentParser(description="获取杭州人工认为有缺陷大图")
    parser.add_argument('--data',
                        default=r"creepageDistance.yaml",
                        help="没有分的文件夹")
    parser.add_argument('--datasets_path',
                        default=r"/home/zengxh/datasets/creepageDistance",
                        help="没有分的文件夹")
    opt = parser.parse_args()

    images_dir = os.path.join(opt.datasets_path,"images")
    mkdir(images_dir)

    with open(opt.data,encoding="utf-8") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    train_json = []
    valid_json = []
    for (key, data_dict_tmp) in data_dict["datasets"].items():
        nameImgs = list(paths.list_images(os.path.join(data_dict_tmp["allDatas"])))
        X_train, X_test_val, _, _ = train_test_split(nameImgs, nameImgs, test_size=0.2, random_state=RANDOM_SEED)

        for imagePath in X_train:
            dict_json = get_dict_json(imagePath)
            train_json.append(dict_json)

        for imagePath in X_test_val:
            dict_json = get_dict_json(imagePath)
            valid_json.append(dict_json)

    save_json_file(os.path.join(opt.datasets_path,"train.json"),train_json)
    save_json_file(os.path.join(opt.datasets_path,"valid.json"),valid_json)