# -*- coding: utf-8 -*-
# Time    : 2023/10/30 20:35
# Author  : fanc
# File    : train_base.py

import warnings

warnings.filterwarnings("ignore")
import os.path
import time
import os
import argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
from src.dataloader.load_data import split_data, my_dataloader
from src.models.networks.resnet_add_feature import generate_model
import json
from utils import AverageMeter2 as AverageMeter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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
    model.load_state_dict(state_dict)
    # 如果需要在多GPU上运行模型
    if multi_gpu:
        # 使用DataParallel封装模型
        model = nn.DataParallel(model)

    return model

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
        self.best_metrics = {}
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.args = args
        if args.num_classes > 2:
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.summaryWriter = summaryWriter
        self.scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and self.device.type == 'cuda'))
        self.best_score = float('-inf')
        self.best_score_epoch = 0
        self.no_improve_epochs = 0
        self.monitor_name = "val_f1_macro" if args.num_classes > 2 else "val_accuracy"
        self.self_model()

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
                    improved = self.evaluate()
                    if improved:
                        self.no_improve_epochs = 0
                    else:
                        self.no_improve_epochs += 1
                    if self.no_improve_epochs >= self.args.early_stop_patience:
                        print(f"Early stopping at epoch {self.epoch} (no improvement for {self.no_improve_epochs} evals).")
                        break
        else:
            self.evaluate()

    def self_model(self):
        if self.args.MODEL_WEIGHT:
            self.model = load_model(model=self.model,
                            checkpoint_path=self.args.MODEL_WEIGHT,
                            multi_gpu=torch.cuda.device_count() > 1)
            print('load model weight success!')
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs.")
        self.model.to(self.device)

    def calculate_metrics(self, pred, label):
        with torch.no_grad():
            if self.args.num_classes > 2:
                pred_cls = torch.softmax(pred, dim=1).argmax(dim=1)
                acc = accuracy_score(label, pred_cls)
                precision = precision_score(label, pred_cls, average="macro", zero_division=0)
                recall = recall_score(label, pred_cls, average="macro", zero_division=0)
                f1 = f1_score(label, pred_cls, average="macro", zero_division=0)
            else:
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5).float()
                acc = accuracy_score(label, pred)
                precision = precision_score(label, pred, zero_division=0)
                recall = recall_score(label, pred, zero_division=0)
                f1 = f1_score(label, pred, zero_division=0)
            return acc, precision, recall, f1
    def calculate_all_metrics(self, pred, label):
        if self.args.num_classes > 2:
            pred = torch.tensor(pred)
            pred_cls = torch.softmax(pred, dim=1).argmax(dim=1)
            label = torch.tensor(label).long()
            acc = accuracy_score(label, pred_cls)
            precision = precision_score(label, pred_cls, average="macro", zero_division=0)
            recall = recall_score(label, pred_cls, average="macro", zero_division=0)
            f1 = f1_score(label, pred_cls, average="macro", zero_division=0)
            return acc, precision, recall, f1, 0.0
        pred = torch.sigmoid(torch.tensor(pred))
        pred = (pred > 0.5).float()
        acc = accuracy_score(label, pred)
        precision = precision_score(label, pred, zero_division=0)
        recall = recall_score(label, pred, zero_division=0)
        f1 = f1_score(label, pred, zero_division=0)
        return acc, precision, recall, f1, 0.0

    def get_meters(self):
        meters = {
            'loss': AverageMeter(),'accuracy': AverageMeter(), 'precision': AverageMeter(),
            'recall': AverageMeter(),'f1': AverageMeter()
        }
        return meters
    def update_meters(self, meters, values):
        for meter, value in zip(meters, values):
            meter.update(value)

    def reset_meters(self, meters):
        for meter in meters:
            meter.reset()
    def print_metrics(self, meters, prefix=""):
        metrics_str = ' '.join([f'{k}: {v.avg:.4f}' if isinstance(v, AverageMeter) else f'{k}: {v:.4f}' for k, v in meters.items()])
        print(f'{prefix} {metrics_str}')

    def log_metrics_to_tensorboard(self, metrics, epoch, stage_prefix=''):
        """
        将指标和损失值写入TensorBoard，区分损失和指标，以及训练和验证阶段。
        参数:
        - metrics (dict): 包含指标名称和值的字典。
        - epoch (int): 当前的epoch。
        - stage_prefix (str): 用于区分训练和验证阶段的前缀（如'Train'/'Val'）。
        - category_prefix (str): 用于区分损失和性能指标的前缀（如'Loss'/'Metric'）。
        """
        for name, meter in metrics.items():
            if 'loss' not in name.lower():
                category_prefix = 'Metric'
            else:
                category_prefix = 'Loss'
            tag = f'{category_prefix}/{name}'
            if 'lr' in name.lower():
                tag = 'lr'
            value = meter.avg if isinstance(meter, AverageMeter) else meter
            self.summaryWriter.add_scalars(tag, {stage_prefix: value}, epoch)

    def train_one_epoch(self):
        self.model.train()
        meters = self.get_meters()
        all_preds = []
        all_labels = []
        self.optimizer.zero_grad()
        for inx, (img, mask, label, clinical) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            clinical = clinical.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(self.args.amp and self.device.type == 'cuda')):
                cls = self.model(img, clinical)[-1]
                if self.args.num_classes > 2:
                    y = label.long().view(-1)
                    if cls.dim() != 2:
                        raise ValueError(f"Expected logits shape [B, C] for multiclass, got {tuple(cls.shape)}")
                    c = cls.size(1)
                    y_min = int(y.min().item())
                    y_max = int(y.max().item())
                    if y_min < 0 or y_max >= c:
                        raise ValueError(
                            f"Label index out of range: label range [{y_min}, {y_max}] but model has C={c}. "
                            f"Set --num-classes correctly or check Histology mapping."
                        )
                    loss = self.loss_function(cls, y)
                else:
                    loss = self.loss_function(cls, label)
                loss_for_backward = loss / self.args.accumulation_steps
            self.scaler.scale(loss_for_backward).backward()
            if ((inx + 1) % self.args.accumulation_steps == 0) or ((inx + 1) == len(self.train_loader)):
                if self.args.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            all_preds.extend(cls.detach().cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            acc, precision, recall, f1 = self.calculate_metrics(cls.cpu(), label.cpu())
            self.update_meters(
                [meters[i] for i in meters.keys()],
                [loss, acc, precision, recall, f1])

        meters['accuracy'], meters['precision'], meters['recall'], meters['f1'], meters['auc'] = self.calculate_all_metrics(all_preds, all_labels)
        self.print_metrics(meters, prefix=f'Epoch: [{self.epoch}]{len(self.train_loader)}]')
        self.log_metrics_to_tensorboard(meters, self.epoch, stage_prefix='Train')
        self.log_metrics_to_tensorboard({'lr':self.optimizer.param_groups[0]['lr']}, self.epoch)

    def evaluate(self):
        self.model.eval()  # 切换模型到评估模式
        meters = self.get_meters()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():  # 禁用梯度计算
            for inx, (img, mask, label, clinical) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                img = img.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                clinical = clinical.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(self.args.amp and self.device.type == 'cuda')):
                    cls = self.model(img, clinical)[-1]
                # pred = torch.sigmoid(cls)

                if self.args.num_classes > 2:
                    y = label.long().view(-1)
                    if cls.dim() != 2:
                        raise ValueError(f"Expected logits shape [B, C] for multiclass, got {tuple(cls.shape)}")
                    c = cls.size(1)
                    y_min = int(y.min().item())
                    y_max = int(y.max().item())
                    if y_min < 0 or y_max >= c:
                        raise ValueError(
                            f"Label index out of range: label range [{y_min}, {y_max}] but model has C={c}. "
                            f"Set --num-classes correctly or check Histology mapping."
                        )
                    loss_val = self.loss_function(cls, y)
                else:
                    loss_val = self.loss_function(cls, label)
                # dice_loss_val = self.dice_loss(seg, mask)
                # total_loss_val = self.loss_weight[0] * cls_loss_val + self.loss_weight[1] * dice_loss_val

                all_preds.extend(cls.detach().cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                if self.args.num_classes > 2:
                    all_probs.extend(torch.softmax(cls, dim=1).detach().cpu().numpy())
                else:
                    all_probs.extend(torch.sigmoid(cls).detach().cpu().numpy())
                acc, precision, recall, f1 = self.calculate_metrics(cls.cpu(), label.cpu())
                self.update_meters(
                    [meters[i] for i in meters.keys()],
                    [loss_val, acc, precision, recall, f1])

        meters['accuracy'], meters['precision'], meters['recall'], meters['f1'], _ = self.calculate_all_metrics(all_preds, all_labels)
        meters['monitor'] = meters['f1'] if self.args.num_classes > 2 else meters['accuracy']
        self.print_metrics(meters, prefix=f'Epoch-Val: [{self.epoch}]{len(self.train_loader)}]')
        # 更新学习率调度器
        self.scheduler.step(meters['loss'].avg)
        # 记录性能指标到TensorBoard
        self.log_metrics_to_tensorboard(meters, self.epoch, stage_prefix='Val')
        print(f'Best {self.monitor_name} is {self.best_score} at epoch {self.best_score_epoch}!')
        print(f'{self.best_score}=>{meters["monitor"]}')

        # 检查并保存最佳模型
        if meters['monitor'] > self.best_score:
            self.best_score_epoch = self.epoch
            self.best_score = meters['monitor']
            self.best_metrics = meters
            with open(os.path.join(os.path.dirname(self.args.save_dir), 'best_acc_metrics.json'), 'w')as f:
                json.dump({k: v for k, v in meters.items() if not isinstance(v, AverageMeter)}, f)
            # 保存模型检查点
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_score': self.best_score,
            }, os.path.join(self.args.save_dir, 'best_checkpoint.pth'))
            print(f"New best model saved at epoch {self.best_score_epoch} with {self.monitor_name}: {self.best_score:.4f}")
            improved = True
        else:
            improved = False
        self.print_metrics(meters, prefix=f'Epoch(Val): [{self.epoch}][{inx + 1}/{len(self.train_loader)}]')

        if self.epoch % self.args.save_epoch == 0:
            checkpoint = {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),  # *模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                    'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                    'best_score': meters['monitor'],
                    'num_params': self.num_params
                }
            torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-%d.pth' % self.epoch))
            print(f"New checkpoint saved at epoch {self.epoch} with {self.monitor_name}: {meters['monitor']:.4f}")
        return improved

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def apply_preset(args):
    if args.preset == 't4_12h':
        if args.rd == 50:
            args.rd = 18
        if args.batch_size == 1:
            args.batch_size = 1
        if args.epochs == 100:
            args.epochs = 60
        if args.accumulation_steps == 4:
            args.accumulation_steps = 8
        if args.early_stop_patience == 10:
            args.early_stop_patience = 8
        if args.lr == 0.0001:
            args.lr = 1e-4
        if args.dropout == 0:
            args.dropout = 0.2
        args.amp = True

def main(args, path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    # data
    with open('/kaggle/working/KidneyStoneSC/configs/dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    data_dir = dataset['data_dir']
    infos_name = dataset['infos_name']
    filter_volume = dataset.get('filter_volume', 0.0)
    train_info, val_info = split_data(data_dir, infos_name, filter_volume, rate=0.8, seed=args.seed)
    if train_info:
        detected_num_classes = len(set(int(x["label"]) for x in train_info))
        if detected_num_classes > 1 and args.num_classes != detected_num_classes:
            print(f"[Info] Detected {detected_num_classes} classes from metadata. Overriding --num-classes={args.num_classes}.")
            args.num_classes = detected_num_classes
    train_loader = my_dataloader(data_dir,
                                      train_info,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      phase='train',
                                      pin_memory=device.type == "cuda",
                                      persistent_workers=args.num_workers > 0,
                                      prefetch_factor=args.prefetch_factor)
    clinical_dim = train_loader.dataset.clinical_dim if hasattr(train_loader.dataset, "clinical_dim") else 0
    val_loader = my_dataloader(data_dir,
                                     val_info,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     phase='val',
                                     clinical_preprocessor=getattr(train_loader.dataset, "clinical_preprocessor", None),
                                     pin_memory=device.type == "cuda",
                                     persistent_workers=args.num_workers > 0,
                                     prefetch_factor=args.prefetch_factor)
    model = generate_model(
        model_depth=args.rd,
        n_classes=args.num_classes,
        dropout_rate=args.dropout,
        n_input_features=clinical_dim
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=8)
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
    parser.add_argument('--preset', type=str, default='none', choices=['none', 't4_12h'])
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--rd', type=int, default=50)
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--log_interval', type=int, default=1)
    # parser.add_argument('--input-path', type=str, default='/home/wangchangmiao/kidney/data/')
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--seed', type=int, default=1900)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--early-stop-patience', type=int, default=10)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)

    opt = parser.parse_args()
    apply_preset(opt)
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