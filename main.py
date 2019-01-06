from __future__ import print_function
import argparse
import os
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
#from models import ResNet18
#from models import ResNeXt29_2x64d
import numpy as np
import torch.utils.data as data
from PIL import Image
from util.cutout import Cutout
from models import resnet20_cifar

parser = argparse.ArgumentParser(description='PyTorch CINIC10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#best_acc = 0  # best test accuracy

        
class CINIC10(data.Dataset):
 
    classes =['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root, train=True,transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img = []
        self.lab = []
        
        file_name = self.root
       # print(file_name)
        # now load the numpy arrays
        if os.path.exists(file_name):
            data= np.load(file_name)
            self.img = data['img']
            self.lab = data['label']
        else:
            print("It can't find .np")
        
            #self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            
    def __getitem__(self, index):
       
        img, lab= self.img[index], self.lab[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            lab = self.target_transform(lab)

        return img, lab


    def __len__(self):
        return len(self.img)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    

def train(epoch,net, criterion, optimizer,trainset,trainloader):
    ''' Trains the net on the train dataset for one entire iteration '''
    print('\nEpoch:' , (int(epoch)+1))
    #global best_acc
    net.train()
    train_loss = 0
    correct = 0
    #total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        #total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


    print('Train: total accuracy is: %.2f %% , loss = %.3f \n' % ( 100 * correct / len(trainset), train_loss/len(trainset) ))
    

# def validate(epoch,net, criterion,validset, validloader):
#     ''' Validates the net's accuracy on validation dataset and saves if better accuracy than previously seen. '''
#     #global best_acc
#     net.eval()
#     valid_loss = 0
#     correct = 0
#    # total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(validloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#             valid_loss += loss.item()
#             _, predicted = outputs.max(1)
#         #        total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#     print('Val: total accuracy is: %.2f %% , loss = %.3f \n' % ( 100 * correct / len(validset), valid_loss/len(validset)) )


#     # Save checkpoint.
#     acc = 100.*correct/len(validset)
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt3.t7')
#         best_acc = acc


def test(net, criterion,optimizer,testset,testloader,epoch,scheduler,best_acc):
    ''' Final test of the best performing net on the testing dataset. '''
    #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #checkpoint = torch.load('./checkpoint/ckpt.t7')
    #net.load_state_dict(checkpoint['net'])
    #global best_acc

    net.eval()
    test_loss = 0
    correct = 0
   # total = 0
   #print('Test best performing net from epoch {} with accuracy {:.3f}%'.format(checkpoint['epoch'], checkpoint['acc']))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Test: total accuracy is: %.2f %% , loss = %.3f \n' % ( 100 * correct / len(testset), test_loss/len(testset)) )
    # Save checkpoint.
    acc = 100.*correct/len(testset)
    if acc > best_acc:
        print('Saving..')
        state = {
            'state_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_last-V4_all.t7')
        best_acc = acc
        
        state_2 = {
            'state_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state_2, './checkpoint/ckpt_last-V4.t7')

def main():
    
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    total_epoch = 400  # The number of total epochs to run

    # Data
    print('==> Preparing data..')

    cinic_directory = '/tmp/dataset-nctu/CINIC-10/'
    train_dir = './train_img.npz'
    test_dir = './test_img.npz'

    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    #cinic_mean = [0.4789, 0.4723, 0.4305]
    #cinic_std = [0.1998, 0.1965, 0.1997]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
        #Cutout(n_holes=1, length=16)
        #Cutout(n_holes=1, length=16)
    ])

    #trainset = torchvision.datasets.ImageFolder(root=(cinic_directory + '/train'), transform=train_transform)
    trainset  = CINIC10(root=train_dir,transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

    validset = torchvision.datasets.ImageFolder(root=(cinic_directory + '/valid'), transform=transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False, num_workers=4)

    #testset = torchvision.datasets.ImageFolder(root=(cinic_directory + '/test'), transform=transform)
    testset = CINIC10(root=test_dir,transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    classes = ('airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    #net = VGG('VGG16')
    #net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    
    net = resnet20_cifar()
    net = net.to(device)
    
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        
  
    
   
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=total_epoch, eta_min=0)
    
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt_last-V4_all.t7')
        net.load_state_dict(checkpoint['state_dict'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(start_epoch," - best_acc = " , best_acc)
        
    else:
        best_acc = 0
    start_time = datetime.now()
    print('Runnning training and test for {} epochs'.format(total_epoch))

    # Run training for specified number of epochs
    for epoch in range(start_epoch, start_epoch+total_epoch):
        scheduler.step()
        train(epoch,net, criterion, optimizer,trainset, trainloader)
        #validate(epoch,net, criterion,validset, validloader)
        test(net, criterion, optimizer, testset,testloader,epoch,scheduler,best_acc)

    time_elapsed = datetime.now() - start_time
    print('Training time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))

    # Run final test on never before seen data
    #test(net, criterion,  testset,testloader)
 
if __name__ == '__main__':
    main()