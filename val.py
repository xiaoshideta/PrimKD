import os
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from config.nyu_b4 import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from models.builder import EncoderDecoder2 as segmodel2
from models.builder import EncoderDecoder3 as segmodel3
from dataloader.dataloader_nyu import ValPre

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device, flag):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        if flag == "rgb":
            # print("rgb: ", flag)
            pred = self.sliding_eval_rgbX(img, None, config.eval_crop_size, config.eval_stride_rate, device)
        elif flag == 'depth':
            # print("depth: ", flag)
            pred = self.sliding_eval_rgbX(modal_x, None, config.eval_crop_size, config.eval_stride_rate, device)
        else:
            # print("rgbd: ", flag)
            pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)

        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line, mIoU = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)
        return result_line, mIoU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--lambda_mask', type=float, default=0.75, help='Description of new argument')
    parser.add_argument('--decode_init', type=int, default=0, help='Description of new argument')
    parser.add_argument('--losses', nargs='+', default=['loss1','loss2','loss3','loss4'], help='Names of the losses to be used')

    args = parser.parse_args()
    device = str(0)
    all_dev = parse_devices(device)
    print(config.backbone)
    network = segmodel3(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d, load=True , decode_init=0, losses=args.losses, lambda_mask=args.lambda_mask)
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        rgb_mIoU = segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file, config.link_val_log_file, None, "rgbd")
        print('rgb_mIoU: %.3f%%' % (rgb_mIoU))