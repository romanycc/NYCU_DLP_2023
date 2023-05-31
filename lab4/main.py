from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from dataloader import RetinopathyLoader
from dataloader import getData
import torchvision.models as models
import csv

acc_train = []
acc_test = []

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1,1)),
        )
        
        self.avepool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 5)
        # create a list of sequential modules
        modules = nn.ModuleList([self.conv1,
                                Add_Basic(64,64,(1,1),False), 
                                Add_Basic(64,128,(2,2),True),
                                Add_Basic(128,256,(2,2),True),
                                Add_Basic(256,512,(2,2),True)
        ])
        # concatenate all sequential modules into a single sequential module
        self.model = nn.Sequential(*modules)
        #print(self.model)

    def forward(self, x):
        x = self.model(x)           #print(x.shape) (num, 512, n, n)
        x = self.avepool(x)         #print(x.shape) (num, 512, 1, 1)
        x = x.view(x.shape[0], -1)  #print(x.shape) (num, 512)
        output = self.fc(x)         #print(x.shape) (num, 5)
        return output

class Basicblock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=(1,1), downsample=False):
        super(Basicblock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=strides, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if downsample:
            self.ds = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(2,2), bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.ds = nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.model(x)
        tmp = self.ds(x)
        output = self.relu2( out + tmp )
        return output

def Add_Basic(in_channel, out_channel, stride, downsample):
    model = []
    model.append(Basicblock(in_channel, out_channel, stride, downsample))
    model.append(Basicblock(out_channel, out_channel, (1,1), downsample=False))
    return nn.Sequential(*model)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1,1))
        )
        # create a list of sequential modules
        modules = nn.ModuleList([self.conv1,
                                Add_Bottle(64,256,64,3,(1,1)), 
                                Add_Bottle(256,512,128,4,(2,2)),
                                Add_Bottle(512,1024,256,6,(2,2)),
                                Add_Bottle(1024,2048,512,3,(2,2))
        ])
        # concatenate all sequential modules into a single sequential module
        self.model = nn.Sequential(*modules)
        self.avepool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 5)

    def forward(self, x):
        x = self.model(x)
        x = self.avepool(x)
        x = x.view(x.shape[0], -1)
        output = self.fc(x)
        return output

class Bottleneck(nn.Module):
    def __init__(self, last_channel,in_channel, out_channel,  strides=(1,1), downsample=False):
        super(Bottleneck, self).__init__()
        if downsample:
            self.model = nn.Sequential(
                nn.Conv2d(last_channel, out_channel, kernel_size=(1,1), stride=(1,1), bias=False),
                nn.BatchNorm2d(out_channel),
                nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=strides, padding=(1,1), bias=False),
                nn.BatchNorm2d(out_channel),
                nn.Conv2d(out_channel, in_channel, kernel_size=(1,1), stride=(1,1), bias=False),
                nn.BatchNorm2d(in_channel)
            )
            self.ds = nn.Sequential(
                nn.Conv2d(last_channel, in_channel, kernel_size=(1,1), stride=strides, bias=False),
                nn.BatchNorm2d(in_channel)
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1), bias=False),
                nn.BatchNorm2d(out_channel),
                nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
                nn.BatchNorm2d(out_channel),
                nn.Conv2d(out_channel, in_channel, kernel_size=(1,1), stride=(1,1), bias=False),
                nn.BatchNorm2d(in_channel)
            )
            self.ds = nn.Identity()



        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.model(x)
        tmp = self.ds(x)
        output = self.relu2( out + tmp )
        return output

def Add_Bottle(last_channel, in_channel, out_channel, block_num, stride):
    model = []
    model.append(Bottleneck(last_channel, in_channel, out_channel, stride, downsample=True))
    for i in range(block_num-1):
        model.append(Bottleneck(last_channel, in_channel, out_channel))
    return nn.Sequential(*model)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    global acc_test
    acc_test.append(100. * correct / len(test_loader.dataset))


#Test the training set
def test_train(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            target = target.type(torch.LongTensor)
            data, target = data.to(device,dtype=torch.float32), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()  # sum up batch loss
            # print(output.shape) (1000,2) (80,2) 
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability of each row
            correct += pred.eq(target.view_as(pred)).sum().item() #pred compare to target

    test_loss /= len(test_loader.dataset)

    print('\nTrain set: Average loss: {:.10f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    global acc_train
    acc_train.append(100. * correct / len(test_loader.dataset))



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
        print("========cuda========")
    elif use_mps:
        device = torch.device("mps")
        print("========mps========")
    else:
        device = torch.device("cpu")
        print("========cpu=======")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = RetinopathyLoader(root = "train/",mode="train")
    dataset2 = RetinopathyLoader(root = "test/",mode="test")

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    #by hand model
    model = ResNet18().to(device)
    # model = ResNet50().to(device)
    #by web model
    #model = models.resnet18(weights='IMAGENET1K_V1')
    # model = models.resnet50(pretrained=True)

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 5)
    # model.to(device)


    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_train(model, device, train_loader)
        test(model, device, test_loader)
        #scheduler.step()

    with open('acc_test_50_pre.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(acc_test)
    
    with open('acc_train_50_pre.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(acc_train)

    if args.save_model:
        torch.save(model.state_dict(), "Resnet50_pretrained.pt")


if __name__ == '__main__':
    main()