import torch
import torch.nn as nn
from tqdm import tqdm
import os
from dataloader import train_loader, val_loader
from model import unet
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

num_classes = 8

def train(train_loader, model, criterion, optimizer):
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    model.train()

    for input, target in tqdm(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.sum().item()
        train_acc_sum += (output.argmax(dim=1) == target).float().sum().item()
        n += target.shape[0]

    return train_loss_sum / n, train_acc_sum / (n*256*256)

def validate(val_loader, model, criterion):
    val_loss_sum, val_acc_sum, n = 0.0, 0.0, 0
    model.eval()
    for input, target in tqdm(val_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target.long())

        val_loss_sum += loss.sum().item()
        val_acc_sum += (output.argmax(dim=1) == target).float().sum().item()
        n += target.shape[0]

    return val_loss_sum / n, val_acc_sum / (n*256*256)

model = unet(num_classes=num_classes).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

best_loss = 2020.0
epochs = 1
train_loss_l = []
val_loss_l = []
train_acc_l = []
val_acc_l = []

for epoch in range(epochs):
    print('[Info]: Start epoch ', epoch)
    train_loss, train_acc = train(train_loader, model, criterion, optimizer)
    val_loss, val_acc = validate(val_loader, model, criterion)
    # scheduler.step()

    train_loss_l.append(train_loss)
    train_acc_l.append(train_acc)
    val_loss_l.append(val_loss)
    val_acc_l.append(val_acc)

    print('Epoch %d: train loss %.4f, train acc %.3f, val_loss %.4f, val_acc %.3f'
          % (epoch, train_loss, train_acc, val_loss, val_acc))
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './baseline.pt')




