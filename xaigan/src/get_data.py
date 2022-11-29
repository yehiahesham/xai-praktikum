from torch.utils.data import DataLoader, sampler

from coco_dataset_construction import COCODetection
data_folder = "./data"

def get_loader(batchSize=100, percentage=1, dataset="mscoco"):
    
    if dataset == "mscoco":
        val_info = r"D:\Praktikum\datasets\coco\annotations_trainval2017\annotations\instances_val2017.json"
        val_image = r"D:\Praktikum\datasets\coco\val2017\val2017"
        
        data = COCODetection(val_image, val_info)
        indices = [i for i in range(int(percentage * len(data)))]
        loader = DataLoader(data, batch_size=batchSize, sampler=sampler.SubsetRandomSampler(indices))
        return loader
    else:
        raise Exception("dataset name not correct (or not implemented)")
    
