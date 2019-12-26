import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.config import arguments
from model.model import BaseNet
from dataset.dataset import FlowerData


def train(model, dataloader, criterion, device, optimizer):
    model.train()

    epoch_loss = 0
    epoch_accuracy = 0

    for data, target in dataloader:
        data = data.to(device).half()
        target = target.to(device).long()
        output = model(data)

        _, pred = torch.max(output, 1)
        epoch_accuracy += (pred == target).sum().item()

        loss = criterion(output, target)
        epoch_loss += loss.item() * dataloader.batch_size

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss /= len(dataloader.dataset)
    epoch_accuracy /= len(dataloader.dataset)

    return epoch_loss, epoch_accuracy


def test(model, dataloader, criterion, device):
    model.eval()

    epoch_loss = 0
    epoch_accuracy = 0

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device).half()
            target = target.to(device).long()
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item() * dataloader.batch_size
            _, pred = torch.max(output, 1)
            epoch_accuracy += (pred == target).sum().item()

    epoch_loss /= len(dataloader.dataset)
    epoch_accuracy /= len(dataloader.dataset)

    return epoch_loss, epoch_accuracy


def main(args, round_num):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dataset_train = FlowerData(args, split='train')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    dataset_test = FlowerData(args, split='test')
    dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=8)

    model = BaseNet(num_class=args.class_num)
    model = model.to(device).half()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer.zero_grad()

    criterion = nn.CrossEntropyLoss()
    criterion.to(device).half()
    train_total_loss = {}
    train_total_acc = {}

    test_total_loss = {}
    test_total_acc = {}

    all_epochs = tqdm(range(args.epochs))

    for epoch in all_epochs:
        # train
        train_epoch_loss, train_epoch_acc = \
            train(model, dataloader_train, criterion, device, optimizer)

        train_total_loss[epoch + 1] = train_epoch_loss
        train_total_acc[epoch + 1] = train_epoch_acc

        # basic info
        info = {'epoch': '%d/%d' % (epoch + 1, args.epochs)}

        # train stat
        if (epoch + 1) % args.log_interval == 0:
            info['train loss'] = '%.2f' % train_epoch_loss
            info['train accuracy'] = '%.4f' % train_epoch_acc

        # test stat
        if (epoch + 1) % args.test_interval == 0:
            test_epoch_loss, test_epoch_acc= \
                test(model, dataloader_test, criterion, device)

            test_total_loss[epoch + 1] = test_epoch_loss
            test_total_acc[epoch + 1] = test_epoch_acc

            info['test loss'] = '%.2f' % test_epoch_loss
            info['test accuracy'] = '%.4f' % test_epoch_acc

        # all stat
        all_epochs.set_postfix(info)

        # save stat
        base_path = './log/round'
        np.save(base_path + '%.2d_train_total_loss' % round_num, train_total_loss)
        np.save(base_path + '%.2d_train_total_acc' % round_num, train_total_acc)
        np.save(base_path + '%.2d_test_total_loss' % round_num, test_total_loss)
        np.save(base_path + '%.2d_test_total_acc' % round_num, test_total_acc)

        # save model
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), './checkpoint/round%.2d_epoch%.4d.pth' % (round_num, epoch + 1))

# model = BaseNet(num_class=args.class_num)
# model.load_state_dict(torch.load('./checkpoint/base_epoch%.4d.pth' % 10))


if __name__ == '__main__':
    argument = arguments()

    for i in range(10):
        print('round: %d' % i)
        main(argument, i)

