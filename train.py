import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config.nyu_b4 import config
# from config2 import config
from dataloader.dataloader_nyu import get_train_loader, ValPre
from models.builder import EncoderDecoder as segmodel
from models.builder import EncoderDecoder2 as segmodel2
from models.builder import EncoderDecoder3 as segmodel3

from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, ensure_dir, link_file, load_model, parse_devices
from utils.metric import hist_info, compute_score
from engine.evaluator import Evaluator
from utils.visualize import print_iou, show_img
from tensorboardX import SummaryWriter
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--distillation_alpha', type=float, default=1, help='Description of new argument')
parser.add_argument('--distillation_beta', type=float, default=0.1, help='Description of new argument')
parser.add_argument('--distillation_single', type=int, default=1, help='Description of new argument')
parser.add_argument('--mask_single', type=str, default='hint', help='Description of the string variable')
parser.add_argument('--distillation_flag', type=int, default=0, help='Description of new argument')
parser.add_argument('--lambda_mask', type=float, default=0.75, help='Description of new argument')
parser.add_argument('--select', type=str, default='max', help='Description of the string variable')
parser.add_argument('--decode_init', type=int, default=0, help='Description of new argument')
parser.add_argument('--losses', nargs='+', default=['loss1','loss2','loss3','loss4'], help='Names of the losses to be used')
logger = get_logger()
os.environ['MASTER_PORT'] = '169710'


class KLDivergenceCalculator():
    def __init__(self):
        pass

    def softmax(self, logits):
        return F.softmax(logits, dim=1)

    def compute_kl_divergence(self, logits_p, logits_q):
        prob_p = self.softmax(logits_p)
        prob_q = self.softmax(logits_q)
        log_prob_p = F.log_softmax(logits_p, dim=1)
        log_prob_q = F.log_softmax(logits_q, dim=1)
        kl_div = torch.sum(prob_p * (log_prob_p - log_prob_q), dim=1)
        return kl_div.mean()

def distill_feature_maps(rgbd_features, rgb_features):
    mse_loss = nn.MSELoss()
    total_loss = 0
    loss = mse_loss(rgbd_features, rgb_features)
    total_loss += loss
    return total_loss

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
                                      val_dataset.class_names, show_no_back=False)
        return result_line, mIoU

class Record(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)
    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
        "class_names": config.class_names,
    }
    val_pre = ValPre()
    val_dataset = RGBXDataset(data_setting, 'val', val_pre)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M-%S", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
        path3 = tb_dir + '/exp.log'
        sys.stdout = Record(path3, sys.stdout)
    print(args)

    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    criterion2 = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    kl_calculator = KLDivergenceCalculator()

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
        BatchNorm2d2 = nn.BatchNorm2d
    else:
        BatchNorm2d = nn.BatchNorm2d
        BatchNorm2d2 = nn.BatchNorm2d

    if args.mask_single == "mask_hint":
        model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d, load=True, decode_init=0, losses=args.losses, lambda_mask=args.lambda_mask)
    else:
        model = segmodel3(cfg=config, criterion=criterion, norm_layer=BatchNorm2d, load=True, decode_init=0, losses=args.losses, lambda_mask=args.lambda_mask)

    config.backbone = 'single_'+config.backbone
    print(config.backbone)
    model2 = segmodel2(cfg=config, criterion=criterion2, norm_layer=BatchNorm2d2, load=True, decode_init=1)
    for param in model2.parameters():
        param.requires_grad = False

    base_lr = config.lr
    # base_lr2 = config.lr
    params_list = []
    # params_list2 = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    # params_list2 = group_weight(params_list2, model2, BatchNorm2d2, base_lr2)


    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                            output_device=engine.local_rank, find_unused_parameters=False)
            device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model2.to(device1)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model2.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer, model2=model2, optimizer2=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()

    logger.info('begin trainning:')
    Best_IoU = 0.0
    Best_rgb_IoU = 0.0
    Best_cmx_IoU = 0.0
    Best_depth_IoU = 0.0
    if args.distillation_flag == 1:
        print("use (teacher.detach,student)")
    if args.distillation_single == 1:
        print("use loss_rdkl")
    print("use_loss:", args.losses)
    print("distillation_alpha:", args.distillation_alpha)
    print("distillation_beta:", args.distillation_beta)
    print("lambda_mask:", args.lambda_mask)
    print("select_method:", args.select)
    print("method:", args.mask_single)

    for epoch in range(engine.state.epoch, config.nepochs + 1):
        model.train()
        model2.eval()
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        sum_loss = 0
        sum_loss2 = 0
        sum_kl_loss = 0
        for idx in pbar:
            engine.update_iteration(epoch, idx)
            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']
            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)
            aux_rate = 0.2
            logits, rgbd_x, loss = model(imgs, modal_xs, gts)
            logits2, rgb_x, loss2 = model2(imgs, None, gts)
            if args.distillation_flag:
                loss_rdkl = kl_calculator.compute_kl_divergence(logits2.detach(), logits) * args.distillation_alpha
            else:
                loss_rdkl = kl_calculator.compute_kl_divergence(logits, logits2.detach()) * args.distillation_alpha
            if args.distillation_single == 1:
                loss = loss + loss_rdkl
            else:
                loss = loss
            feature_loss = 0.0
            loss_values = {
                'loss1': [],
                'loss2': [],
                'loss3': [],
                'loss4': []
            }
            num_values = 0
            for loss_name in args.losses:
                
                loss_values[loss_name].append(distill_feature_maps(rgbd_x[num_values], rgb_x[int(loss_name[-1])-1].detach()))
                num_values = num_values + 1
            selected_losses = args.losses
            selected_loss_values = [loss_values[loss_name][-1] for loss_name in selected_losses]

            selected_loss_tensor = torch.stack(selected_loss_values)
            max_loss_value = torch.max(selected_loss_tensor)
            min_loss_value = torch.min(selected_loss_tensor)
            feature_loss = 0
            for loss_value in selected_loss_values:
                if args.select == 'max':
                    feature_loss = torch.where(torch.eq(selected_loss_tensor, max_loss_value), selected_loss_tensor * args.distillation_beta, selected_loss_tensor * 0.0).sum()
                elif args.select == 'min':
                    feature_loss = torch.where(torch.eq(selected_loss_tensor, min_loss_value), selected_loss_tensor * args.distillation_beta, selected_loss_tensor * 0.0).sum()
                else:
                    feature_loss = torch.where(torch.eq(selected_loss_tensor, max_loss_value), selected_loss_tensor * 0.0, selected_loss_tensor * 0.0).sum()
            loss = loss + feature_loss
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
                reduce_loss2 = all_reduce_tensor(loss2, world_size=engine.world_size)
                reduce_kl_loss = all_reduce_tensor(loss_rdkl, world_size=engine.world_size)
                reduce_middle_loss = all_reduce_tensor(feature_loss, world_size=engine.world_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                sum_loss2 += reduce_loss2.item()
                sum_kl_loss += reduce_kl_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1))) \
                            + ' loss2=%.4f total_loss2=%.4f' % (reduce_loss2.item(), (sum_loss2 / (idx + 1))) \
                            + ' kl_loss=%.4f' % (reduce_kl_loss.item()) \
                            + ' middle_loss=%.4f' % (reduce_middle_loss.item())
            else:
                sum_loss += loss
                sum_loss2 += reduce_loss2
                sum_kl_loss += reduce_kl_loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1))) \
                            + ' loss2=%.4f total_loss2=%.4f' % (reduce_loss2.item(), (sum_loss2 / (idx + 1)))

            del loss
            del loss2
            del feature_loss
            del loss_rdkl
            del max_loss_value
            del min_loss_value
            pbar.set_description(print_str, refresh=False)

        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
        network = None
        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (
                epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                model.eval()
                model2.eval()
                device = str(0)
                all_dev = parse_devices(device)
                with torch.no_grad():
                    segmentor = SegEvaluator(val_dataset, config.num_classes, config.norm_mean,
                                             config.norm_std, network,
                                             config.eval_scale_array, config.eval_flip,
                                             all_dev, verbose=False, save_path=None,
                                             show_image=False)
                    config.val_log_file = tb_dir + '/val_' + '.log'
                    config.link_val_log_file = tb_dir + '/val_last.log'
                    config.checkpoint_dir = tb_dir + '/checkpoint'
                    rgb_mIoU = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                         config.link_val_log_file, model, "rgbd")
                    if (Best_rgb_IoU < rgb_mIoU):
                        Best_rgb_IoU = rgb_mIoU
                        engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                        config.log_dir,
                                                        config.log_dir_link,
                                                        Best_rgb_IoU, Best_depth_IoU)
                        print("save successful!")
                    depth_mIoU = 0.0
                    print('epoch: %d, rgbd_mIoU: %.3f%%, Best_rgbd_IoU: %.3f%%' % (epoch, rgb_mIoU, Best_rgb_IoU))

            elif not engine.distributed:
                model.eval()
                device = '0'
                all_dev = parse_devices(device)
                with torch.no_grad():
                    segmentor = SegEvaluator(val_dataset, config.num_classes, config.norm_mean,
                                             config.norm_std, network,
                                             config.eval_scale_array, config.eval_flip,
                                             all_dev, verbose=False, save_path=None,
                                             show_image=False)
                    config.val_log_file = tb_dir + '/val_' + '.log'
                    config.link_val_log_file = tb_dir + '/val_last.log'
                    config.checkpoint_dir = tb_dir + '/checkpoint'
                    mIoU = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                         config.link_val_log_file, model)
                    print('epoch: %d, mIoU: %.3f%%, Best_IoU: %.3f%%' % (epoch, mIoU, Best_IoU))
                    if (Best_IoU < mIoU):
                        Best_IoU = mIoU
                        engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                        config.log_dir,
                                                        config.log_dir_link,
                                                        Best_IoU)
                        print("save successful!")
