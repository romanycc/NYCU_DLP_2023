import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
from torchvision import datasets, transforms

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv', names=["obj"])  #####
        label = pd.read_csv('train_label.csv', names=["obj"])
        img = np.squeeze(img.values)
        img = np.delete(img, [15663,15970,17616])
        label = np.squeeze(label.values)
        label = np.delete(label, [15663,15970,17616])
        print(img.shape,label.shape)
        return img, label
    else:
        img = pd.read_csv('test_img.csv', names=["obj"])
        label = pd.read_csv('test_label.csv', names=["obj"])
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))
        #self.check()

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = self.root + self.img_name[index] + '.jpeg'
        width, height = Image.open(path).size

        if self.mode == 'train':
            transform=transforms.Compose([
                transforms.CenterCrop(height),
                transforms.Resize(512),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            transform=transforms.Compose([
                transforms.CenterCrop(height),
                transforms.Resize(512),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        img = transform(Image.open(path).convert('RGB'))
        label = self.label[index]
        return img, label
    
    def check(self):
        for index in range(7026):
            print(index)
            path = self.root + self.img_name[index] + '.jpeg'
            label = self.label[index]
            try:
                with Image.open(path) as img:
                    try:
                        img = img.convert('RGB')
                        # Do something with the image here
                    except PIL.UnidentifiedImageError:
                        print('Error: Unidentified image at', path,index)
                    except Exception:
                        print('Error: Failed to convert image at', path,index)
            except OSError:
                print('Error: Failed to open file at', path,index)
            except Exception:
                print('Error: Unknown error occurred while processing', path,index)
