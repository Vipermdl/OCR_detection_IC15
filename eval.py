#!/usr/local/bin/python
# -*-coding=utf-8 -*-
import argparse
import torch
import sys
import os
import logging
import pathlib
import traceback

from tqdm import tqdm

from model.model import FOTSModel
from utils.bbox import Toolbox

logging.basicConfig(level=logging.DEBUG, format='')


def load_model(model_path, with_gpu):
    logger.info("Loading checkpoint: {} ...".format(model_path))
    checkpoints = torch.load(model_path)
    if not checkpoints:
        raise RuntimeError('No checkpoint found.')
    config = checkpoints['config']
    state_dict = checkpoints['state_dict']

    model = FOTSModel(config['model'])

    model.load_state_dict(state_dict)

    model = torch.nn.DataParallel(model)

    if with_gpu:
        model = model.cuda()
    model = model.eval()
    return model


def load_annotation(gt_path):
    with gt_path.open(mode='r') as f:
        label = dict()
        label["coor"] = list()
        label["ignore"] = list()
        for line in f:
            text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
            if text[8] == "###" or text[8] == "*":
                label["ignore"].append(True)
            else:
                label["ignore"].append(False)
            bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            label["coor"].append(bbox)
    return label


def main(args: argparse.Namespace):
    model_path = args.model
    image_dir = args.image_dir
    output_img_dir = args.output_img_dir
    output_txt_dir = args.output_txt_dir

    if output_img_dir is not None and not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if output_txt_dir is not None and not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)

    annotation_dir = args.annotation_dir
    with_image = True if output_img_dir else False
    with_gpu = True if torch.cuda.is_available() else False

    model = load_model(model_path, with_gpu)
    if annotation_dir is not None:

        true_pos, true_neg, false_pos, false_neg = [0] * 4
        for image_fn in tqdm(image_dir.glob('*.jpg')):
            gt_path = annotation_dir / image_fn.with_name('gt_{}'.format(image_fn.stem)).with_suffix('.txt').name
            labels = load_annotation(gt_path)
            # try:
            with torch.no_grad():
                polys, im, res = Toolbox.predict(image_fn, model, with_image, output_img_dir, with_gpu, labels,
                                                 output_txt_dir)
            true_pos += res[0]
            false_pos += res[1]
            false_neg += res[2]
        if (true_pos + false_pos) > 0:
            precision = true_pos / (true_pos + false_pos)
        else:
            precision = 0
        if (true_pos + false_neg) > 0:
            recall = true_pos / (true_pos + false_neg)
        else:
            recall = 0
        print("TP: %d, FP: %d, FN: %d, precision: %f, recall: %f" % (true_pos, false_pos, false_neg, precision, recall))
    else:
        with torch.no_grad():
            for image_fn in tqdm(image_dir.glob('*.jpg')):
                Toolbox.predict(image_fn, model, with_image, output_img_dir, with_gpu, None,None)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-m', '--model',
                        default='./model_best.pth.tar',
                        type=pathlib.Path,
                        help='path to model')
    parser.add_argument('-o', '--output_img_dir', type=pathlib.Path,
                        help='output dir for drawn images')
    parser.add_argument('-t', '--output_txt_dir', type=pathlib.Path,
                        help='output dir for drawn images')
    parser.add_argument('-i', '--image_dir', default='/mnt/disk1/dataset/icdar2015/4.4/test/ch4_test_images',
                        type=pathlib.Path,
                        help='dir for input images')
    parser.add_argument('-a', '--annotation_dir',
                        type=pathlib.Path,
                        help='dir for input images')

    args = parser.parse_args()
    main(args)
