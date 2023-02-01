from pycocotools.coco import COCO
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import  json
import scipy.io

class Flowers102_detection(data.Dataset):
    def __init__(self, image_path, labels_path,captions_path,target_image_w=256,target_image_h=256):
        self.root     = image_path
        self.captions = None
        self.labels = scipy.io.loadmat(labels_path)['labels'][0]
        with open(captions_path) as json_file:
            self.captions  = json.load(json_file)
        self.ids   = sorted(list(self.captions.keys()))
        self.target_image_w=target_image_w
        self.target_image_h=target_image_h

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img,label,captions = self.pull_item(index)
        return  img,label,captions

    def pull_item(self, index):
        
        img_id = self.ids[index] # ex: '0001'->[0]

        #Get image by img_id 
        file_name = f'image_{img_id}.jpg'
        path = osp.join(self.root, file_name)

        img = cv2.imread(path)
        img = cv2.resize(img, (self.target_image_w,self.target_image_h))
        

        #Proccess image captions 
        captions = self.captions[img_id]
        label = None
        label = self.labels[index]
        return torch.from_numpy(img).permute(2, 0, 1),label,captions


if __name__=='__main__':
      
    from models.configurations import configurations
    CFG  = configurations["datasets"]["flowers-102"]
    train_image   = CFG["path"]+"/"+CFG["train_images"]
    labels_info   = CFG["path"]+"/"+"imagelabels.mat"
    captions_path = CFG["path"]+"/"+"flowers102_captions.json"
    
    dataset = Flowers102_detection(train_image, labels_info,captions_path)
    loader = DataLoader(dataset)
    for n_batch, (real_batch) in enumerate(loader):
    # for n, (img,captions,file_name) in enumerate(loader):
        #print(n)
        img,label,captions = real_batch #images,class
        img = np.uint8(img.squeeze().numpy())
        #cv2.imshow('img', img)
        #print(file_name)
        #print(captions)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)
        
        
        #gt, masks, num_crowds = label
        #masks = masks.squeeze(0)
        """
        for m in range(masks.size(0)):
            mask = masks[m].numpy()
            color = np.random.randint(0, 255)
            channel = np.random.randint(0, 3)
            y, x = np.where(mask == 1)
            img[y, x, channel] = color
        """
        # cv2.imshow('img', img)
        # cv2.waitKey(500)
        