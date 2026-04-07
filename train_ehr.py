# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import os.path
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
from src.dataloader.load_data import split_data, my_dataloader
import time
import json
from utils import AverageMeter2 as AverageMeter
from utils import calculate_acc_sigmoid
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from src.models.networks.nets import ehr_net

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
        self.best_acc = 0
        self.best_metrics = {}
        self.best_acc_epoch = 0
        self.args = args
        self.loss_function = torch.nn.BCEWithLogitsLoss()# torch.nn.BCELoss()
        self.summaryWriter = summaryWriter
        self.use_clip = False
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
                    self.evaluate()
            self.print_metrics(self.best_metrics, prefix='The Best metrics in epoch {}'.format(self.best_acc_epoch))
        else:
            self.evaluate()

    def self_model(self):
        if self.args.MODEL_WEIGHT:
            self.model = load_model(model=self.model,
                            checkpoint_path=self.args.MODEL_WEIGHT,
                            multi_gpu=torch.cuda.device_count() > 1)
            print('load model weight success!')
        self.model.to(self.device)

    def calculate_metrics(self, pred, label):
        with torch.no_grad():
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            acc = accuracy_score(label, pred)
            precision = precision_score(label, pred)
            recall = recall_score(label, pred)
            f1 = f1_score(label, pred)
            return acc, precision, recall, f1
    def calculate_all_metrics(self, pred, label):
        pred = torch.sigmoid(torch.tensor(pred))
        pred = (pred > 0.5).float()
        acc = accuracy_score(label, pred)
        precision = precision_score(label, pred)
        recall = recall_score(label, pred)
        f1 = f1_score(label, pred)
        auc = roc_auc_score(label, pred)
        return acc, precision, recall, f1, auc

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
        for inx, (img, mask, label, clinical) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            img, label, clinical, mask = img.to(self.device), label.to(self.device), clinical.to(self.device), mask.to(
                self.device)
            cls = self.model(clinical)
            # print(torch.unique(cls), torch.unique(label))
            loss = self.loss_function(cls, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
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
        with torch.no_grad():  # 禁用梯度计算
            for inx, (img, mask, label, clinical) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                img, label, clinical, mask = img.to(self.device), label.to(self.device), clinical.to(
                    self.device), mask.to(self.device)
                cls = self.model(clinical)
                # pred = torch.sigmoid(cls)

                loss_val = self.loss_function(cls, label)
                # dice_loss_val = self.dice_loss(seg, mask)
                # total_loss_val = self.loss_weight[0] * cls_loss_val + self.loss_weight[1] * dice_loss_val

                all_preds.extend(cls.detach().cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                acc, precision, recall, f1 = self.calculate_metrics(cls.cpu(), label.cpu())
                self.update_meters(
                    [meters[i] for i in meters.keys()],
                    [loss_val, acc, precision, recall, f1])

        meters['accuracy'], meters['precision'], meters['recall'], meters['f1'], meters['auc'] = self.calculate_all_metrics(all_preds, all_labels)
        self.print_metrics(meters, prefix=f'Epoch-Val: [{self.epoch}]{len(self.train_loader)}]')
        # 更新学习率调度器
        self.scheduler.step(meters['loss'].avg)
        # 记录性能指标到TensorBoard
        self.log_metrics_to_tensorboard(meters, self.epoch, stage_prefix='Val')
        print(f'Best acc is {self.best_acc} at epoch {self.best_acc_epoch}!')
        print(f'{self.best_acc}=>{meters["accuracy"]}')

        if self.args.phase == 'train':
            # 检查并保存最佳模型
            if meters['accuracy'] > self.best_acc:
                self.best_acc_epoch = self.epoch
                self.best_acc = meters['accuracy']
                self.best_metrics = meters
                with open(os.path.join(os.path.dirname(self.args.save_dir), 'best_acc_metrics.json'), 'w')as f:
                    json.dump({k: v for k, v in meters.items() if not isinstance(v, AverageMeter)}, f)
                # 保存模型检查点
                torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_acc': self.best_acc,
                }, os.path.join(self.args.save_dir, 'best_checkpoint.pth'))
                print(f"New best model saved at epoch {self.best_acc_epoch} with accuracy: {self.best_acc:.4f}")
            self.print_metrics(meters, prefix=f'Epoch(Val): [{self.epoch}][{inx + 1}/{len(self.train_loader)}]')

            if self.epoch % self.args.save_epoch == 0:
                checkpoint = {
                        'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),  # *模型参数
                        'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                        'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                        'best_acc': meters['accuracy'],
                        'num_params': self.num_params
                    }
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-%d.pth' % self.epoch))
                print(f"New checkpoint saved at epoch {self.epoch} with accuracy: {meters['accuracy']:.4f}")

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main(args, path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    # model = generate_model(model_depth=args.rd, n_classes=args.num_classes, dropout_rate=args.dropout)
    model = ehr_net()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    # data
    with open('/kaggle/working/KidneyStoneSC/configs/dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    data_dir = dataset['data_dir']
    infos_name = dataset['infos_name']
    filter_volume = dataset['filter_volume']
    train_info, val_info = split_data(data_dir, infos_name, filter_volume, rate=0.8)
    # with open(os.path.join(data_dir, 'train_clinical_infos.json'), 'r', encoding='utf-8') as f:
    #     train_info = json.load(f)
    # with open(os.path.join(data_dir, 'val_clinical_infos.json'), 'r', encoding='utf-8') as f:
    #     val_info = json.load(f)
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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')


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