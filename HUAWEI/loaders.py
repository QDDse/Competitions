# -*- coding: UTF-8 -*-
import os
import cv2
import torch.utils.data as data
import numpy as np
from PIL import Image
import csv
from itertools import islice
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

## new_Dataset
class Rane_Dataset(data.Dataset):
    '''
    args:
        imagedir: 存放image的地址
        df: Dataframe, train_df or val_df
        tranfrom: 
    '''
    def __init__(self, imagedir, df, transform, require_label=True):
        self.imagedir = imagedir
        self.df = df
        self.require_label = require_label
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        ## img
        # image = Image.open(os.path.join(self.imagedir, line['image']))
        # image = image.convert('RGB')
        img = cv2.imread(os.path.join(self.imagedir, line['image']))
        # img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image = img)['image']  ## Notice!! 
        ## label
        label = line['label_01']
        # print(type(class_label))
        # if class_label == 0:
        #     label = 0
        # else:
        #     label = 1
        if self.require_label:
            # print(type(image))
            return (img, label)
        else:
            return img
        
    def __len__(self):
        return self.df.shape[0]
    
def build_loader_new(
        imagedir,
        batch_size,
        num_workers,
        metafile,
        require_label=True,
        transform = None,
):
    if transform == None:
        trfs = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        trfs = transforms.Compose(trfs)
    else:
        trfs = transform
        
    dataset = Rane_Dataset(imagedir, metafile, trfs, require_label)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


## Old 
class ImageSet(data.Dataset):
    '''
    args:
        imagedir: 
        metadir:
    '''
    def __init__(
            self,
            imagedir,
            metadir,
            metafile,
            transform,
            require_label=True
    ):
        self.imagedir = imagedir
        self.metafile = metafile
        self.transform = transform
        self.require_label = require_label
        
        ## labeled_train_data
        if require_label:
            self.images, self.labels = self.get_txt_info(metadir, metafile)
        ## test_data or unlabeled_data
        else:
            self.images = self.get_unlabled_info(self.imagedir)
            
    def get_unlabled_info(self, imgdir):
        image_ls = os.listdir(imgdir)
        if '.ipynb_checkpoints' in image_ls:
            image_ls.remove('.ipynb_checkpoints')
        return os.listdir(imgdir)
    
    def get_txt_info(self, metadir, metafile):
        images = list()
        labels = list()
        with open(os.path.join(metadir, metafile), 'r', encoding='utf8') as fp:
            reader = csv.reader(fp)
            for row in islice(reader, 1, None):
                imagename = row[0]
                images.append(imagename)
                if self.require_label:
                    class_label = row[1]
                    if class_label == '0':
                        label = 0
                    else:
                        label = 1
                    labels.append(label)
        if self.require_label:
            return images, labels
        else:
            return images, None

    def __getitem__(self, item):
        imagename = self.images[item]
        image = Image.open(os.path.join(self.imagedir, imagename))
        image = image.convert('RGB')
        image = self.transform(image)
        if self.require_label:
            label = self.labels[item]
            return image, label
        else:
            return image, imagename

    def __len__(self):
        return len(self.images)


def build_loader(
        imagedir,
        batch_size,
        num_workers,
        metadir,
        metafile,
        require_label=True
):
    trfs = [
        # transforms.Resize((224, 224)),  # Swin-384
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    trfs = transforms.Compose(trfs)
    dataset = ImageSet(imagedir, metadir, metafile, trfs, require_label)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader



