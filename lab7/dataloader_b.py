import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import json
from sklearn import preprocessing
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch
import numpy as np
import itertools
from torchvision.utils import save_image
import torchvision

def getCode(root):
    path = root + "objects.json"
    with open(path) as file:
        code = json.load(file)
    return code

def getTrainData(root, mode, code):
    path = root + mode + ".json"
    with open(path) as file:
        data = json.load(file)

    lb = preprocessing.LabelBinarizer()
    lb.fit([i for i in range(24)])

    # make data
    img_name = []
    labels = []
    for key, value in data.items():
        img_name.append(key)
        tmp = []
        for i in range(len(value)):
            tmp.append(np.array(lb.transform([code[value[i]]])))
        labels.append((np.sum(tmp, axis=0)))
    print("train_img_name:", len(img_name))
    print("train_labels:", len(labels))
    labels = torch.tensor(np.array(labels))
    #print(labels[0])
    return img_name, labels




def getTestData(root, mode, code):
    path = root + mode + ".json"
    with open(path) as file:
        data = json.load(file)
    lb = preprocessing.LabelBinarizer()
    lb.fit([i for i in range(24)])
    # make data
    labels = []
    for value in data:
        tmp = []
        for i in range(len(value)):
            tmp.append(np.array(lb.transform([code[value[i]]])))
        labels.append(np.sum(tmp, axis=0))
    print("test_labels:", len(labels))
    labels =torch.tensor(np.array(labels))
    #print(labels[3])
    return labels

class iclevrLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.code = getCode(root)
        # get data
        if mode=="train":
            self.img_name, self.label = getTrainData(root, mode, self.code)
        elif mode=="test" or mode=="new_test":
            self.label = getTestData(root, mode, self.code)
        else:
            raise ValueError("No such root!")
        self.mode = mode
        #self.check()

    def __len__(self):
        """'return the size of dataset"""
        return len(self.label)

    def __getitem__(self, index):
        # Return processed image and label

        if self.mode == 'train':
            path = self.root + "iclevr/" +self.img_name[index]
            transform=transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
            img = transform(Image.open(path).convert('RGB'))
            #print(img.shape)
        else:
            img = torch.ones(1) # for sampling, give a dummy values 

        label = self.label[index]
        return img, label
    

def save_images(images, name):
    save_image(images, fp = "./"+name+".png")

l = iclevrLoader("dataset/", "test")
