from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from dataloader import read_bci_data
from torch.utils.data import Dataset
import csv

acc_train = []
acc_test = []

#Inheritance nn.Module to create nn for DeepConvNet
class DeepConvNet(nn.Module):       
    def __init__(self,activation=nn.ReLU()):
        super(DeepConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 25, (1,5)),
            nn.Conv2d(25, 25, (2,1)),
            nn.BatchNorm2d(25),
            activation,
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5),
            nn.Conv2d(25,50,(1,5)),
            nn.BatchNorm2d(50),
            activation,
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5),
            nn.Conv2d(50,100,(1,5)),
            nn.BatchNorm2d(100),
            activation,
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5),
            nn.Conv2d(100,200,(1,5)),
            nn.BatchNorm2d(200),
            activation,
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(8600, 2)

    def forward(self, x):
        x = self.model(x)   #torch.Size([B, 200, 1, 43])
        x = x.view(-1,8600) #torch.Size([B, 8600])
        output = self.fc(x)
        return output

#Inheritance nn.Module to create nn for EEGNet
class EEGNet(nn.Module):       
    def __init__(self,activation=nn.ReLU()):
        super(EEGNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,16,(1,51),stride=(1,1),padding=(0,25),bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32,(2,1),stride=(1,1),groups=16,bias=False),
            nn.BatchNorm2d(32),
            activation,
            nn.AvgPool2d(kernel_size=(1,4),stride=(1,4),padding=0),
            nn.Dropout(0.25),
            nn.Conv2d(32,32,(1,15),stride=(1,1),padding=(0,7),bias=False),
            nn.BatchNorm2d(32),
            activation,
            nn.AvgPool2d(kernel_size=(1,8),stride=(1,8),padding=0),
            nn.Dropout(0.25)
        )
        self.fc = nn.Linear(736, 2, bias=True)

    def forward(self, x):
        x = self.model(x)       #torch.Size([B, 32, 1, 23])
        x = x.view(-1,736)      #torch.Size([B=64, 736])
        output = self.fc(x)
        return output

#A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. 
class EEGDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label

#將模型從評估模式轉為訓練模式
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()               
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        data, target = data.to(device,dtype=torch.float32), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

#將模型從訓練模式轉為評估模式
def test(model, device, test_loader):
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

    print('\nTest set: Average loss: {:.10f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    global acc_test
    acc_test.append(100. * correct / len(test_loader.dataset))
    if (100. * correct / len(test_loader.dataset)) > 87:
        s = "EEGNet_best.pth"
        torch.save(model.state_dict(), s)

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
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
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
    parser.add_argument('--m', type=int, default=1, 
                        help='EEGNet : 1, DeepConvNet : 2')
    parser.add_argument('--a', type=int, default=2, 
                        help='LeakyReLU : 1, ReLU : 2, ELU : 3')
    args = parser.parse_args()
   
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    #manipulate data
    train_data, train_label, test_data, test_label = read_bci_data()
    train_loader = torch.utils.data.DataLoader(EEGDataset(train_data, train_label),**train_kwargs,shuffle=True)
    test_loader = torch.utils.data.DataLoader(EEGDataset(test_data, test_label), **test_kwargs)
    
    #set activation function
    act = {1 : nn.LeakyReLU(), 2 : nn.ReLU(), 3 : nn.ELU()}
    act_name = {1 : "LeakyReLU", 2 : "ReLU", 3 : "ELU"}

    #create model
    if args.m == 1:
        model = EEGNet(activation=act[args.a]).to(device)
        print("Using EEGNet model with ",act_name[args.a]," activation")
    else:
        model = DeepConvNet(activation=act[args.a]).to(device)
        print("Using DeepConvNet model with ",act_name[args.a]," activation")
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    #start training and testing
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        test_train(model, device, train_loader)
        #print(acc_test)
        #print(acc_train)
        #scheduler.step()

    with open('acc_test.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(acc_test)
    
    with open('acc_train.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(acc_train)

    if args.save_model:
        if args.m == 1:
            print("Using EEGNet model with ",act_name[args.a]," activation")
            print(model)
            s = "EEGNet_"+act_name[args.a]+".pth"
            torch.save(model.state_dict(), s)
        else:
            print("Using DeepConvNet model with ",act_name[args.a]," activation")
            print(model)
            s = "DeepConvNet_"+act_name[args.a]+".pth"
            torch.save(model.state_dict(), s)

if __name__ == '__main__':
    main()