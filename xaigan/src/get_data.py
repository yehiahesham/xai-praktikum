from torch.utils.data import DataLoader, sampler
from models.configurations import configurations
from coco_dataset_construction import COCODetection
data_folder = "./data"

def get_loader(batchSize=8, percentage=1, dataset="mscoco"):
    
    if dataset == "mscoco":
        # val_info = r"D:\Documents\TUM\WS2022\XAI\MS_COCO\annotations_trainval2017\annotations\instances_val2017.json"
        # val_image = r"D:\Documents\TUM\WS2022\XAI\MS_COCO\val2017\val2017"
        MS_COCO_CFG  = configurations["datasets"]["MS_COCO"]
        val_image    = MS_COCO_CFG["path"]+"/"+MS_COCO_CFG["val_images"]
        val_info     = MS_COCO_CFG["path"]+"/"+MS_COCO_CFG["trainval_annotations"]+"/"+"instances_val2017.json"
        val_cap      = MS_COCO_CFG["path"]+"/"+MS_COCO_CFG["trainval_annotations"]+"/"+"captions_val2017.json"
        data = COCODetection(val_image, val_info,val_cap)

        indices = [i for i in range(int(percentage * len(data)))]
        loader = DataLoader(data, batch_size=batchSize, sampler=sampler.SubsetRandomSampler(indices), collate_fn=lambda x: x)
        return loader
    else:
        raise Exception("dataset name not correct (or not implemented)")
    
