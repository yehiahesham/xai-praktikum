import os



configurations= {   
    "datasets":{
        "MS_COCO":{
            "path": "C:/Users/caner/Desktop/MS_COCO",
            # "path":"/home/yehia/Documents/MS_COCO",
            # "path":"/home2/yehia.ahmed/XAI_datasets/MS-COCO",
            "train_images__path":"root",
            "test_image_info_path": "image_info_test2017/annotations",
            "trainval_annotations": "annotations_trainval2017/annotations",
            "val_images"  : "val2017/val2017",
            "train_images": "train2017/train2017",
            "test_images" : "test2017/test2017",
            "classes": ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
        },
        "flowers-102":{
            # "path": "C:/Users/caner/Desktop/MS_COCO",
            # "path":"/home/yehia/Documents/MS_COCO",
            "path": "/home2/yehia.ahmed/XAI_datasets/data/flowers-102",
            "train_images": "jpg",
            "val_images"  : "val",
            "test_images" : "test",
        },
        "others":{
            "Datasets_DownloadPATH":"/home2/yehia.ahmed/XAI_datasets",
            }
    
    }
}