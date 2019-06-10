import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import numpy as np
import argparse
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.cuda as cuda
import torchvision
import collections
from models.LENet import *
from models.VGG import *
from models.resnet import *
from models.mobileNet import *
from models.ShuffleNet import *
from resnext import *

def parse_arg():
    parse = argparse.ArgumentParser(description='the argument to config the training')
    parse.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parse.add_argument('--resume', '-r', default=None, help='restore the model from file')
    parse.add_argument('-gpu', default = True, action='store_true', help='use gpu')
    parse.add_argument('--work_dir', default='./work_dir_ShuffleNet', help='the directory save the model')
    parse.add_argument('--max_epoch', default=200, help='the number of epoch to train')

    args = parse.parse_args()
    return args

cfg = parse_arg()
best_acc = 0
batch_size = 128
epoch = 0
if cfg.gpu == True:
    device = 'cuda' if cuda.is_available() else 'cpu'
else:
    device = 'cpu'

print('preparing the data....')
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', download=True,
                                                 train=True, transform=train_transform)
train_loader = data.DataLoader(train_dataset, batch_size =batch_size, shuffle=True, num_workers=4)
test_dataset = torchvision.datasets.CIFAR10(root='./data',download = True,
                                                train = False, transform = test_transform)
test_loader = data.DataLoader(test_dataset, batch_size = batch_size, num_workers= 4)

#model = LENET()
#model = resnet101()
#model = MobileNetv1()
#model = ShuffleNetG3()
#model = VGG('vgg19')
model = ResNext101()
optimizer = optim.SGD(model.parameters(), lr= cfg.lr, momentum=0.9, weight_decay=0.0001)
lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, [110, 160 ], 0.1)
criterion = nn.CrossEntropyLoss()


if device == 'cuda':
    cudnn.benchmark = True
    model = nn.DataParallel(model)


if cfg.resume is not None:
    resume_path = cfg.resume
    print('loading state dict...')
    assert os.path.isdir(cfg.resume)
    state_dict = torch.load(os.path.join(cfg.resume, 'ckpt.pth'))
    model.load_state_dict(state_dict['net'])
    epoch = state_dict['epoch']
    best_acc = state_dict['best_acc']


def train(epoch):
    print('epoch{}\n'.format(epoch))
    lr_schedule.step()
    train_loss = 0
    correct = 0
    num = 0
    model.train()
    for i, (input,target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        result = model(input)
        optimizer.zero_grad()
        loss = criterion(result,target)
        loss.backward()
        optimizer.step()

        _,prediction = result.max(1)
        correct += prediction.eq(target).sum().item()
        train_loss +=loss
        num += target.size(0)
        if i%100==0:
            print('epooch {} {}/{}: train accuracy: {}, train loss: {}'.format(epoch,i, len(train_loader),
                                                                                correct/num,train_loss/num))
    print('epooch {} : train accuracy: {}, train loss: {}'.format(epoch, correct / num, train_loss / num))
def test(epoch):
    global best_acc
    model.eval()
    test_loss= 0
    correct = 0
    num = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)
            result = model(input)
            loss = criterion(result, target)
            _,prediction = result.max(1)
            correct += prediction.eq(target).sum().item()
            test_loss += loss
            num+=target.size(0)
            if i%100==0:
                print('epoch {}|{}/{}: test accuray: {}, test loss: {}'.format(epoch,i,len(test_loader),
                                                                               correct/num,test_loss/num))
        print('epooch {} : train accuracy: {}, test loss: {}'.format(epoch, correct/num, test_loss/num))
        acc = correct/len(test_loader)

        state = {
            'net':model.state_dict(),
            'best_acc':acc,
            'epoch': epoch,
                }
        print('save the model for accuray: {}'.format(correct/num))
        save_path = cfg.work_dir
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    if best_acc < acc:
        best_acc=acc
        torch.save(state, os.path.join(save_path, 'bestckpt.pth'))
    else:
        torch.save(state, os.path.join(save_path, 'ckpt.pth'))

if __name__=='__main__':
    while epoch<cfg.max_epoch:
        train(epoch)
        test(epoch)
        epoch+=1






