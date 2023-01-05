from torch.utils.data import DataLoader, sampler
from models.configurations import configurations
from coco_dataset_construction import COCODetection
data_folder = "./data"

def get_loader(batchSize=100, percentage=1, dataset="mscoco"):
    
    if dataset == "mscoco":        
        MS_COCO_CFG  = configurations["datasets"]["MS_COCO"]
        
        val_image    = f'{MS_COCO_CFG["path"]}/{MS_COCO_CFG["val_images"]}'
        val_info     = f'{MS_COCO_CFG["path"]}/{MS_COCO_CFG["trainval_annotations"]}/instances_val2017.json'
        val_cap      = f'{MS_COCO_CFG["path"]}/{MS_COCO_CFG["trainval_annotations"]}/captions_val2017.json'

        train_image    = f'{MS_COCO_CFG["path"]}/{MS_COCO_CFG["train_images"]}'
        train_info     = f'{MS_COCO_CFG["path"]}/{MS_COCO_CFG["trainval_annotations"]}/instances_train2017.json'
        train_cap      = f'{MS_COCO_CFG["path"]}/{MS_COCO_CFG["trainval_annotations"]}/captions_train2017.json'


        # test_image    = f'{MS_COCO_CFG["path"]}/{MS_COCO_CFG["test_images"]}'
        # test_info     = f'{MS_COCO_CFG["path"]}/{MS_COCO_CFG["test_image_info_path"]}/instances_val2017.json' ??
        # test_cap      = f'{MS_COCO_CFG["path"]}/{MS_COCO_CFG["test_image_info_path"]}/captions_val2017.json'  ??


        data = COCODetection(val_image, val_info,val_cap)
        # data = COCODetection(train_image, train_info,train_cap)

        indices = [i for i in range(int(percentage * len(data)))]
        loader = DataLoader(data, batch_size=batchSize, sampler=sampler.SubsetRandomSampler(indices), collate_fn=lambda x: x)
        return loader
    else:
        raise Exception("dataset name not correct (or not implemented)")
    
