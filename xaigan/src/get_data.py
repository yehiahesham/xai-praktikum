from torch.utils.data import DataLoader, sampler
from models.configurations import configurations
from coco_dataset_construction import COCODetection
data_folder = "./data"

def get_loader(batchSize=100, percentage=1, dataset="mscoco"):
    
    if dataset == "mscoco":
        # val_info = r"D:\Documents\TUM\WS2022\XAI\MS_COCO\annotations_trainval2017\annotations\instances_val2017.json"
        # val_image = r"D:\Documents\TUM\WS2022\XAI\MS_COCO\val2017\val2017"
        MS_COCO_CFG = configurations["datasets"]["MS_COCO"]
        val_info     = MS_COCO_CFG["path"]+"/"+MS_COCO_CFG["trainval_annotations"]+"/"+"instances_val2017.json"
        val_image    = MS_COCO_CFG["path"]+"/"+MS_COCO_CFG["val_images"]
        
        
        data = COCODetection(val_image, val_info)
        indices = [i for i in range(int(percentage * len(data)))]
        loader = DataLoader(data, batch_size=batchSize, sampler=sampler.SubsetRandomSampler(indices))
        return loader
    else:
        raise Exception("dataset name not correct (or not implemented)")
    
