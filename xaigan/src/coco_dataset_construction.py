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

val_info = r"D:\Praktikum\datasets\coco\annotations_trainval2017\annotations\instances_val2017.json"
val_image = r"D:\Praktikum\datasets\coco\val2017\val2017"
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',*
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')


class COCODetection(data.Dataset):
    def __init__(self, image_path, info_file, has_gt=True):
        self.root = image_path
        self.coco = COCO(info_file)
        self.ids = list(self.coco.imgToAnns.keys())  # 标签数

        if len(self.ids) == 0 or not has_gt:  # 如果没有标签或者不需要GT，则直接使用image
            self.ids = list(self.coco.imgs.keys())
        
        

        self.has_gt = has_gt

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        im, gt, h, w, num_crowds = self.pull_item(index)
        return im   #, (gt, num_crowds)

    def pull_item(self, index):
        img_id = self.ids[index]
        #print(img_id)
        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            #print(target[0])
        else:
            target = []
        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        #print(crowd)
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        #print(target)
        #print('````````````````````````````````````````````````````````````````````')
        num_crowds = len(crowd)

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd
        #print(self.coco.loadImgs(img_id))
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        path = osp.join(self.root, file_name)
        img = cv2.imread(path)
        img = cv2.resize(img, (256,256))
        #print(img.shape)
        height, width, _ = img.shape

        #if len(target) > 0: 
        #masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
        #print(mask)
        #masks = np.vstack(masks)
        #masks = masks.reshape(-1, height, width)
        
        return torch.from_numpy(img).permute(2, 1, 0), target,  height, width, num_crowds


if __name__=='__main__':
    dataset = COCODetection(val_image, val_info)
    loader = DataLoader(dataset)
    for img, label in loader:
        print(img.shape)
        img = np.uint8(img.squeeze().numpy().transpose(1, 0, 2))
        
        gt, masks, num_crowds = label
        masks = masks.squeeze(0)
        """
        for m in range(masks.size(0)):
            mask = masks[m].numpy()
            color = np.random.randint(0, 255)
            channel = np.random.randint(0, 3)
            y, x = np.where(mask == 1)
            img[y, x, channel] = color
        """
        #cv2.imshow('img', img)
        #cv2.waitKey(500)