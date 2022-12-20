from pycocotools.coco import COCO
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

class COCODetection(data.Dataset):
    def __init__(self, image_path, info_file,val_cap, has_gt=True):
        self.root       = image_path
        self.coco       = COCO(info_file)
        self.coco_caps  = COCO(val_cap)
        self.ids        = list(self.coco.imgToAnns.keys())  # 标签数

        if len(self.ids) == 0 or not has_gt:  # 如果没有标签或者不需要GT，则直接使用image
            self.ids = list(self.coco.imgs.keys())
        
        

        self.has_gt = has_gt

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        im ,gt, captions,file_name, h, w, num_crowds = self.pull_item(index)

        return im.float(),(captions,file_name)

    def pull_item(self, index):
        
        img_id = self.ids[index]
        if self.has_gt:
            #load COCO GT values
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)

            #load all captions forimage_id 
            ann_cap_ids = self.coco_caps.getAnnIds(imgIds=img_id)
            ann_cap = self.coco_caps.loadAnns(ann_cap_ids)

        else:
            target = []
        
        #Proccess GT values
        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)
        target += crowd # This is so we ensure that all crowd annotations are at the end of the array

        #Get image by img_id 
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        path = osp.join(self.root, file_name)

        img = cv2.imread(path)
        img = cv2.resize(img, (256,256))
        height, width, _ = img.shape

        #if len(target) > 0: 
        #masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
        #masks = np.vstack(masks)
        #masks = masks.reshape(-1, height, width)
        
        #Proccess image captions 
        captions = []
        for ann in ann_cap:
            captions.append(ann['caption'])
                                           #(1, 0, 2)? 
        return torch.from_numpy(img).permute(1, 0, 2), target,captions,file_name, height, width, num_crowds
                                           #(?, ?, ?)


if __name__=='__main__':
      
    from models.configurations import configurations
    MS_COCO_CFG  = configurations["datasets"]["MS_COCO"]
    val_image    = MS_COCO_CFG["path"]+"/"+MS_COCO_CFG["val_images"]
    val_info     = MS_COCO_CFG["path"]+"/"+MS_COCO_CFG["trainval_annotations"]+"/"+"instances_val2017.json"
    val_cap      = MS_COCO_CFG["path"]+"/"+MS_COCO_CFG["trainval_annotations"]+"/"+"captions_val2017.json"
    
    dataset = COCODetection(val_image, val_info,val_cap)
    loader = DataLoader(dataset)
    for n, (img,captions,file_name) in enumerate(loader):
        #print(n)
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
        