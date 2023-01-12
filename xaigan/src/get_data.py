from torch.utils.data import DataLoader, sampler
from models.configurations import configurations
from coco_dataset_construction import COCODetection
from flowers102_dataset_construction import Flowers102_detection
import torchvision
import torchvision.transforms as transforms

data_folder = "./data"

def get_loader(batchSize=100, percentage=1, dataset="mscoco",target_image_w=32,target_image_h=32):
    download_path = configurations["datasets"]["others"]["Datasets_DownloadPATH"]+"/data"

    if dataset == "mscoco":        
        CFG  = configurations["datasets"]["MS_COCO"]
        
        Classes      = f'{CFG["path"]}/{CFG["classes"]}'
        val_image    = f'{CFG["path"]}/{CFG["val_images"]}'
        val_info     = f'{CFG["path"]}/{CFG["trainval_annotations"]}/instances_val2017.json'
        val_cap      = f'{CFG["path"]}/{CFG["trainval_annotations"]}/captions_val2017.json'

        train_image    = f'{CFG["path"]}/{CFG["train_images"]}'
        train_info     = f'{CFG["path"]}/{CFG["trainval_annotations"]}/instances_train2017.json'
        train_cap      = f'{CFG["path"]}/{CFG["trainval_annotations"]}/captions_train2017.json'


        # test_image    = f'{CFG["path"]}/{CFG["test_images"]}'
        # test_info     = f'{CFG["path"]}/{CFG["test_image_info_path"]}/instances_val2017.json' ??
        # test_cap      = f'{CFG["path"]}/{CFG["test_image_info_path"]}/captions_val2017.json'  ??

        data = COCODetection(val_image, val_info,val_cap,target_image_w=target_image_w,target_image_h=target_image_h)

        indices = [i for i in range(int(percentage * len(data)))]
        loader = DataLoader(data, batch_size=batchSize, sampler=sampler.SubsetRandomSampler(indices), collate_fn=lambda x: x)
        return loader
    elif dataset == "cifar-10":
        transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=download_path, train=True, download=True, transform=transform)
        valset = torchvision.datasets.CIFAR10(root=download_path, train=False, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batchSize, shuffle=True)
        val_loader   = DataLoader(valset, batch_size=batchSize, shuffle=True )
        return train_loader
    elif dataset == "cifar-100":
        transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR100(root=download_path, train=True, download=True, transform=transform)
        valset   = torchvision.datasets.CIFAR100(root=download_path, train=False, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batchSize, shuffle=True)
        val_loader   = DataLoader(valset, batch_size=batchSize, shuffle=True )
        return train_loader
    elif dataset == "flowers-102":
        CFG  = configurations["datasets"]["flowers-102"]
        train_image   = CFG["path"]+"/"+CFG["train_images"]
        labels_info   = CFG["path"]+"/"+"imagelabels.mat"
        captions_path = CFG["path"]+"/"+"flowers102_captions.json"
    
        #for dowonload purposes only 
        trainset = torchvision.datasets.Flowers102(root=download_path, split="train", download=True) 
        # trainset = torchvision.datasets.Flowers102(root=download_path, split="val", download=True) 
        # trainset = torchvision.datasets.Flowers102(root=download_path, split="test", download=True) 
        
        trainset = Flowers102_detection(train_image, labels_info,captions_path,target_image_w=32,target_image_h=32)
        train_loader = DataLoader(trainset, batch_size=batchSize, shuffle=True)
        return train_loader
    
    else:
        raise Exception("dataset name not correct (or not implemented)")
    
