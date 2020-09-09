import torch
import torch.nn as nn
from tqdm import tqdm
import os
from dataloader import train_loader, val_loader
from model import unet
from utils.metrics import SegmentationMetric
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

num_classes = 8
metric = SegmentationMetric(num_classes)

def train(train_loader, model, criterion, optimizer):
    train_loss_sum, train_fwiou_sum, n = 0.0, 0.0, 0
    model.train()

    for input, target in tqdm(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i in range(output.shape[0]):
            pre = output[i, :, :, :].argmax(axis=0).cpu()
            label = target[i, :, :].cpu()
            metric.addBatch(pre, label)
            FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
            train_fwiou_sum += FWIoU

        train_loss_sum += loss.sum().item()
        # train_acc_sum += (output.argmax(dim=1) == target).float().sum().item()
        n += target.shape[0]

    return train_loss_sum / n, train_fwiou_sum / n

def validate(val_loader, model, criterion):
    val_loss_sum, val_fwiou_sum, n = 0.0, 0.0, 0
    model.eval()
    for input, target in tqdm(val_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target.long())

        for i in range(output.shape[0]):
            pre = output[i, :, :, :].argmax(axis=0).cpu()
            label = target[i, :, :].cpu()
            metric.addBatch(pre, label)
            FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
            val_fwiou_sum += FWIoU

        val_loss_sum += loss.sum().item()
        # val_acc_sum += (output.argmax(dim=1) == target).float().sum().item()
        n += target.shape[0]

    return val_loss_sum / n, val_fwiou_sum / n

model = unet(num_classes=num_classes).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

best_loss = 2020.0
epochs = 1

for epoch in range(epochs):
    train_loss, train_FWIoU = train(train_loader, model, criterion, optimizer)
    val_loss, val_FWIoU = validate(val_loader, model, criterion)
    # scheduler.step()

    print('Epoch %d: train loss %.4f, train FWIoU %.3f, val_loss %.4f, val_FWIoU %.3f'
          % (epoch, train_loss, train_FWIoU, val_loss, val_FWIoU))
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './baseline.pt')




