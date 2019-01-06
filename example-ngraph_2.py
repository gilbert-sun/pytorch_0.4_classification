from benchmark import benchmarking
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
import os
import sys
#data_dir = os.environ['TESTDATADIR']
#assert data_dir is not None, "No data directory"
#from models import ResNeXt29_2x64d
from collections import OrderedDict
import time
import numpy as np
from models import resnet20_cifar
import torch.utils.data as data
from PIL import Image

import onnx
import ngraph as ng
from ngraph_onnx.onnx_importer.importer import import_onnx_model

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
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
       
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


BATCH_SIZE = 960

ONNX_MODEL = "./checkpoint/f1_model.onnx"

im_h = 32
im_w = 32
#ngraph
onnx_protobuf = onnx.load(ONNX_MODEL)
ng_model = import_onnx_model(onnx_protobuf)[0]
runtime = ng.runtime(backend_name='CPU')
resnet = runtime.computation(ng_model['output'], *ng_model['inputs'])


model = resnet20_cifar()
inputs = []
targets = []

    
test_dir = './test_img.npz'

cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]

transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=cinic_mean, std=cinic_std)])

    
testset = CINIC10(root=test_dir,transform=transform)
test_c_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,drop_last=True)
test_g_loader = torch.utils.data.DataLoader(testset, batch_size=9600, shuffle=False, num_workers=4,drop_last=True)

@benchmarking(team=7, task=0, model=model, preprocess_fn=None)
def inference(model,**kwargs):
    total = 0
    correct = 0
    assert kwargs['device'] != None, 'Device error'
    device = kwargs['device']
    if device == "cpu":
        print("device = ",device)
        #for inputs, targets in testloader:
        for idx in range(len(test_c_loader)):
            inputs[idx], targets[idx] = inputs[idx].to(device).numpy(), targets[idx].to(device).numpy()
            outputs = resnet(inputs[idx])
            pred = np.argmax(outputs,axis=1)
            total += len(targets[idx])
            if(targets[idx].shape != pred.shape):
                correct += np.equal(targets[idx],pred[0:len(targets)]).sum()
            else:
                correct += np.equal(targets[idx],pred).sum()
            #print("correct=" ,correct )
    else:
        print("device = ",device)
        model.to(device)
        checkpoint = torch.load('./checkpoint/ckpt_last-V4.t7')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (input_, target_) in enumerate(test_g_loader):
                input_, target_ = input_.to(device), target_.to(device)
                output_ = model(input_)
                _, predicted = output_.max(1)
                total += target_.size(0)
                correct += predicted.eq(target_).sum().item()
                #print("correct=",correct)
            

    #total = len(test_loader) * BATCH_SIZE 
    acc = 100.*correct/total
    print(acc)
    return acc





if __name__=='__main__':

    
    for ins, ts in test_c_loader:
        inputs.append(ins)
        targets.append(ts)
    inference(model)