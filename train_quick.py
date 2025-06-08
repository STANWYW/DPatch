import os
import torch
import datetime

from darknet import Darknet19

from datasets.pascal_voc import VOCDataset
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
from random import randint

from patch import *

# 快速训练版本：只训练少量batches
QUICK_TRAIN_BATCHES = 10  # 只训练10个batch进行测试

# data loader
imdb = VOCDataset(cfg.imdb_train, cfg.DATA_DIR, cfg.train_batch_size,
                  yolo_utils.preprocess_train, processes=2, shuffle=True,
                  dst_size=cfg.multi_scale_inp_size)
print('load data succ...')

net = Darknet19()
pretrained_model = 'yolo-voc.weights.h5'

net_utils.load_net(pretrained_model, net, with_patch=False) 
# net.cuda()  # CPU模式注释掉
net.train()
print('load net succ...')

# optimizer
start_epoch = 0
lr = 1e-2
optimizer = torch.optim.SGD([net.patch], lr=lr, momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)

batch_per_epoch = imdb.batch_per_epoch
train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0
t = Timer()
step_cnt = 0
size_index = 0

print(f"快速训练模式：只训练 {QUICK_TRAIN_BATCHES} 个batches")
print("开始训练...")

for step in range(QUICK_TRAIN_BATCHES):
    t.tic()
    
    if step == 0:
        print('-----------save initial patch ------------')
        save_patch(net.patch, step)

    # batch
    batch = imdb.next_batch(size_index)
    im = batch['images']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    dontcare = batch['dontcare']
    orgin_im = batch['origin_im']
    
    # reset patch class here 
    for g, gt_cls in enumerate(gt_classes):
       for c,the_cls in enumerate(gt_cls):
           gt_classes[g][c] = cfg.target_class
    
    # forward
    im_data = net_utils.np_to_variable(im,
                                       is_cuda=False,  # CPU模式
                                       volatile=False).permute(0, 3, 1, 2)
    bbox_pred, iou_pred, prob_pred = net(im_data, gt_boxes, gt_classes, dontcare, size_index)

    # backward
    loss = net.loss
    bbox_loss += net.bbox_loss.data.cpu()
    iou_loss += net.iou_loss.data.cpu()
    cls_loss += net.cls_loss.data.cpu()
    train_loss += loss.data.cpu()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cnt += 1
    step_cnt += 1
    duration = t.toc()
    
    print(f'batch {step+1}/{QUICK_TRAIN_BATCHES}, loss: {loss.data.cpu():.3f}, '
          f'bbox_loss: {net.bbox_loss.data.cpu():.3f}, '
          f'iou_loss: {net.iou_loss.data.cpu():.3f}, '
          f'cls_loss: {net.cls_loss.data.cpu():.3f} '
          f'({duration:.2f} s/batch)')
    
    t.clear()

# 保存最终的patch
print('-----------save final patch ------------')
save_patch(net.patch, "final")

print("快速训练完成！")
print("查看训练的patch:")
print("ls -la trained_patch/1/")

imdb.close() 