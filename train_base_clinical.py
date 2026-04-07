# -*- coding: utf-8 -*-
# Time    : 2023/10/30 20:35
# Author  : fanc
# File    : train_base.py

import warnings

import pandas as pd

warnings.filterwarnings("ignore")
import logging  # 引入logging模块
import os.path
import time
import os
import math
import argparse

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from src.dataloader.load_data import split_data, my_dataloader
from torch.nn.parallel import DataParallel
from src.models.networks.resnet_add_feature import generate_model
import time
import json
import torch.nn.functional as F
from utils import AverageMeter2 as AverageMeter
from utils import calculate_acc_sigmoid
from sklearn.metrics import precision_score, recall_score, f1_score


def load_model(model, checkpoint_path, multi_gpu=False):
    """
    通用加载模型函数。

    :param model: 要加载状态字典的PyTorch模型。
    :param checkpoint_path: 模型权重文件的路径。
    :param multi_gpu: 布尔值，指示是否使用多GPU加载模型。
    :return: 加载了权重的模型。
    """
    # 加载状态字典
    pretrain = torch.load(checkpoint_path)
    if 'model_state_dict' in pretrain.keys():
        state_dict = pretrain['model_state_dict']
    else:
        state_dict = pretrain['state_dict']
    # 检查是否为多卡模型保存的状态字典
    if list(state_dict.keys())[0].startswith('module.'):
        # 移除'module.'前缀（多卡到单卡）
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    # 加载状态字典
    # model.load_state_dict(state_dict)
    # 加载状态字典
    for name, param in model.named_parameters():
        if name in state_dict and param.size() == state_dict[name].size():
            param.data.copy_(state_dict[name])
            # print(f"Loaded layer: {name}")
        else:
            print(f"Skipped layer: {name}")
    # 如果需要在多GPU上运行模型
    if multi_gpu:
        # 使用DataParallel封装模型
        model = nn.DataParallel(model)
    # model.fc = nn.Linear(model.fc.in_features, 16)
    # num_features = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_features, 16),
    #     nn.Linear(16, 1)  # num_classes为您的数据集类别数
    # )

    return model
def precision(pred, label):
    # 计算精确度
    pred = (pred > 0.5).float()
    return precision_score(label.cpu(), pred.cpu())

def recall(pred, label):
    # 计算召回率
    pred = (pred > 0.5).float()
    return recall_score(label.cpu(), pred.cpu())

def calculate_f1_score(pred, label):
    pred = (pred > 0.5).float()
    return f1_score(label.cpu(), pred.cpu())


class Trainer:
    def __init__(self, model, optimizer, device, train_loader, test_loader, scheduler, args, summaryWriter):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = args.epochs
        self.epoch = 0
        self.best_acc = 0
        self.args = args
        self.self_model()
        self.loss_function = torch.nn.BCELoss()
        self.summaryWriter = summaryWriter
        self.use_clip = False
        if args.clinical:
            self.use_clinical = True
            # self.clinical = pd.read_excel(args.clinical)

    def get_result(self):
        return

    def __call__(self):
        if self.args.phase == 'train':
            for epoch in tqdm(range(self.epochs)):
                start = time.time()
                self.epoch = epoch+1
                self.train_one_epoch()
                self.num_params = sum([param.nelement() for param in self.model.parameters()])
                # self.scheduler.step()
                end = time.time()
                print("Epoch: {}, train time: {}".format(epoch, end - start))
                if epoch % 1 == 0:
                    self.evaluate()
        else:
            self.model.eval()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()
            f1_scores = AverageMeter()
            end_time = time.time()
            output_result = []
            import pandas as pd
            with torch.no_grad():
                for inx, (x, mask, label, clinical) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                    # if clinical == 0:
                    #     continue
                    data_time.update(time.time() - end_time)
                    # x = torch.mul(x, mask)
                    x = x.to(self.device)
                    label = label.to(self.device)
                    clinical = clinical.to(self.device)

                    out = self.model(x, clinical)[-1]
                    pred = torch.sigmoid(out)
                    for i in range(pred.size(0)):
                        output_result.append({'pred': pred[i], 'label': label[i], 'id': f'{inx}_{i}'})
                    precision_val = precision(pred, label)
                    recall_val = recall(pred, label)
                    f1_score_val = calculate_f1_score(pred, label)
                    # print('out:{}, sigmoid out:{}, label:{}'.format(out, pred, label))
                    acc = calculate_acc_sigmoid(pred, label)
                    loss = self.loss_function(pred, label)
                    # update
                    losses.update(loss.item(), x.size(0))
                    accuracies.update(acc, x.size(0))
                    precisions.update(precision_val, x.size(0))
                    recalls.update(recall_val, x.size(0))
                    f1_scores.update(f1_score_val, x.size(0))
                    batch_time.update(time.time() - end_time)
                    end_time = time.time()
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc {acc.val:.3f} ({acc.avg:.3f})'
                          '\nout:{out}-pred:{pred}-label:{label}'.format(
                            self.epoch,
                            inx + 1,
                            len(self.test_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            acc=accuracies,
                            out='out',pred='pred', label='label'))
                    print(f'Precision {precisions.val:.3f} ({precisions.avg:.3f})\t'
                          f'Recall {recalls.val:.3f} ({recalls.avg:.3f})\t'
                          f'F1 Score {f1_scores.val:.3f} ({f1_scores.avg:.3f})')
                # df = pd.DataFrame(output_result)
                # df.to_csv('output.csv', index=False)

    def self_model(self):
        if self.args.MODEL_WEIGHT:
            self.model = load_model(model=self.model,
                            checkpoint_path=self.args.MODEL_WEIGHT,
                            multi_gpu=torch.cuda.device_count() > 1)
            print('load model weight success!')
        self.model.to(self.device)

    def evaluate(self):
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end_time = time.time()
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        f1_scores = AverageMeter()

        with torch.no_grad():
            for inx, (x, mask, label, clinical) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                # if clinical == 0:
                #     continue
                data_time.update(time.time() - end_time)
                # x = torch.mul(x, mask)
                x = x.to(self.device)
                label = label.to(self.device)
                clinical = clinical.to(self.device)
                out = self.model(x, clinical)[-1]
                pred = torch.sigmoid(out)

                precision_val = precision(pred, label)
                recall_val = recall(pred, label)
                f1_score_val = calculate_f1_score(pred, label)
                acc = calculate_acc_sigmoid(pred, label)
                loss = self.loss_function(pred, label)
                # update
                accuracies.update(acc, x.size(0))
                precisions.update(precision_val, x.size(0))
                recalls.update(recall_val, x.size(0))
                f1_scores.update(f1_score_val, x.size(0))
                losses.update(loss.item(), x.size(0))
                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    self.epoch,
                    inx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accuracies))
                print(f'Precision {precisions.val:.3f} ({precisions.avg:.3f})\t'
                      f'Recall {recalls.val:.3f} ({recalls.avg:.3f})\t'
                      f'F1 Score {f1_scores.val:.3f} ({f1_scores.avg:.3f})')
            self.scheduler.step(losses.avg)
            self.summaryWriter.add_scalars("Loss", {'Test': losses.avg}, self.epoch)
            self.summaryWriter.add_scalars("Acc", {'Test': accuracies.avg}, self.epoch)
            self.summaryWriter.add_scalars("Precision", {'Test': precisions.avg}, self.epoch)
            self.summaryWriter.add_scalars("Recall", {'Test': recalls.avg}, self.epoch)
            self.summaryWriter.add_scalars("F1 Score", {'Test': f1_scores.avg}, self.epoch)



            if self.epoch % self.args.save_epoch == 0:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),  # *模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                    'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                    'best_val_score': accuracies.avg,
                    'num_params': self.num_params,
                    'epoch': self.epoch
                }
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-%d.pth' % self.epoch))

            if self.best_acc < accuracies.avg:
                self.best_acc = accuracies.avg
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),  # *模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                    'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                    'best_val_score': self.best_acc,
                    'num_params': self.num_params,
                    'epoch': self.epoch
                }
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'best_checkpoint.pth'))


    def train_one_epoch(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        f1_scores = AverageMeter()
        self.model.train()
        end_time = time.time()
        for inx, (x, mask, label, clinical) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            # if clinical == 0:
            #     continue
            data_time.update(time.time() - end_time)
            # input stone
            # x = torch.mul(x, mask)
            x = x.to(self.device)
            label = label.to(self.device)
            clinical = clinical.to(self.device)

            out = self.model(x, clinical)[-1]
            pred = torch.sigmoid(out)
            # print(clinical)
            # print('out, pred and label:', out, pred, label)
            precision_val = precision(pred, label)
            recall_val = recall(pred, label)
            f1_score_val = calculate_f1_score(pred, label)
            acc = calculate_acc_sigmoid(pred, label)
            loss = self.loss_function(pred, label)
            # update
            accuracies.update(acc, x.size(0))
            precisions.update(precision_val, x.size(0))
            recalls.update(recall_val, x.size(0))
            f1_scores.update(f1_score_val, x.size(0))

            losses.update(loss.item(), x.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(self.epoch,
                                                             inx + 1,
                                                             len(self.train_loader),
                                                             batch_time=batch_time,
                                                             data_time=data_time,
                                                             loss=losses,
                                                             acc=accuracies))
            print(f'Precision {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  f'Recall {recalls.val:.3f} ({recalls.avg:.3f})\t'
                  f'F1 Score {f1_scores.val:.3f} ({f1_scores.avg:.3f})')

        self.summaryWriter.add_scalars("Loss", {'Train': losses.avg}, self.epoch)
        self.summaryWriter.add_scalars("Acc", {'Train': accuracies.avg}, self.epoch)
        self.summaryWriter.add_scalars("Precision", {'Train': precisions.avg}, self.epoch)
        self.summaryWriter.add_scalars("Recall", {'Train': recalls.avg}, self.epoch)
        self.summaryWriter.add_scalars("F1 Score", {'Train': f1_scores.avg}, self.epoch)
        self.summaryWriter.add_scalar('Lr', self.optimizer.param_groups[0]['lr'], self.epoch)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main(args, path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    model = generate_model(model_depth=args.rd, n_classes=args.num_classes, dropout_rate=args.dropout)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    # optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    # data
    with open('/kaggle/working/KidneyStoneSC/configs/dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    data_dir = dataset['data_dir']
    infos_name = dataset['infos_name']
    filter_volume = dataset['filter_volume']
    # train_info, val_info = split_data(data_dir, infos_name, filter_volume, rate=0.8)
    with open(os.path.join(data_dir, 'train_clinical_infos.json'), 'r', encoding='utf-8') as f:
        train_info = json.load(f)
    with open(os.path.join(data_dir, 'val_clinical_infos.json'), 'r', encoding='utf-8') as f:
        val_info = json.load(f)
    train_loader = my_dataloader(data_dir,
                                      train_info,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers)
    val_loader = my_dataloader(data_dir,
                                     val_info,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers)
    summaryWriter = None
    if args.phase == 'train':

        log_path = makedirs(os.path.join(path, 'logs'))
        model_path = makedirs(os.path.join(path, 'models'))
        args.log_dir = log_path
        args.save_dir = model_path
        summaryWriter = SummaryWriter(log_dir=args.log_dir)
    trainer = Trainer(model,
                      optimizer,
                      device,
                      train_loader,
                      val_loader,
                      scheduler,
                      args,
                      summaryWriter)
    trainer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rd', type=int, default=50)
    parser.add_argument('--save-epoch', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--clinical', type=bool, default=True)
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--dropout', type=float, default=0)

    opt = parser.parse_args()
    args_dict = vars(opt)
    now = time.strftime('%y%m%d%H%M', time.localtime())
    path = None
    if opt.phase == 'train':
        if not os.path.exists(f'./results/{now}'):
            os.makedirs(f'./results/{now}')
        path = f'./results/{now}'
        with open(os.path.join(path, 'train_config.json'), 'w') as fp:
            json.dump(args_dict, fp, indent=4)
        print(f"Training configuration saved to {now}")
    print(args_dict)

    main(opt, path)