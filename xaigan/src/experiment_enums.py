from enum import Enum

from models.generators import EmbeddingEncoderNetMSCOCO
from models.generators import GeneratorNet_MSCOCO,GeneratorNet_2_MSCOCO,Generator_Encoder_Net_MSCOCO,GeneratorNet_2_EncoderNet_MSCOCO
from models.generators import GeneratorNetCIFAR10,Generator_Encoder_Net_CIFAR10
from models.discriminators import DiscriminatorNetCIFAR10,DiscriminatorNetMSCOCO,DiscriminatorNetMSCOCO_2
from models.text_embedding_models import RobertaClass,Glove_Embbeding

from torch import nn, optim
from experiment import Experiment


class ExperimentEnums(Enum):

    # flowers_only = {
    #     "explainable"     :False,
    #     "explanationType" :None,
    #     "noise_emb_sz"    :100,            #GeneratorNetMSCOCO's noise param
    #     "text_emb_sz"     :768,    #TODO:  #RobertaClass's param
    #     "text_max_len"    :350,            #RobertaClass's param
    #     "use_CLS_emb"     :False,          #RobertaClass's param
    #     "use_one_caption" :True ,          #RobertaClass's param + param used in experiment 
    #     "use_captions"    :False ,          
                            
    #     "generator"    : GeneratorNetCIFAR10,
    #     "discriminator": DiscriminatorNetCIFAR10, 
    #     "text_emb_model":RobertaClass,
    #     "EmbeddingEncoder":None,
        
    #     "dataset": 'flowers-102',  #['flowers-102', 'mscoco', 'cifar-10', 'cifar-100',]
    #     "target_image_w":32,
    #     "target_image_h":32,
    #     "batchSize": 16,#128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 2
    # }
    
    # flowers_Roberta = {
    #     "explainable"     :False,
    #     "explanationType" :None,
    #     "noise_emb_sz"    :100,            #GeneratorNetMSCOCO's noise param
    #     "text_emb_sz"     :768,    #TODO:  #RobertaClass's param
    #     "text_max_len"    :350,            #RobertaClass's param
    #     "use_CLS_emb"     :False,          #RobertaClass's param
    #     "use_one_caption" :True ,          #RobertaClass's param + param used in experiment 
    #     "use_captions"    :True ,          
                            
    #     "generator"    : Generator_Encoder_Net_CIFAR10,
    #     "discriminator": DiscriminatorNetCIFAR10, 
    #     "text_emb_model":RobertaClass,
    #     "EmbeddingEncoder":None,
        
    #     "dataset": 'flowers-102',  #['flowers-102', 'mscoco', 'cifar-10', 'cifar-100',]
    #     "target_image_w":32,
    #     "target_image_h":32,
    #     "batchSize": 16,#128,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 2
    # }
    
    flowers_Glove_2 = {
        "explainable"     :False,
        "explanationType" :None,
        "noise_emb_sz"    :100,            #GeneratorNetMSCOCO's noise param
        "text_emb_sz"     :50,         #Glove param word emb-> sent emb
        'Encoder_emb_sz'  :100+50,
        "text_max_len"    :350,            #Glove param
        "use_CLS_emb"     :False,          #Glove param
        "use_one_caption" :True ,          #Glove param + param used in experiment 
        "use_captions"    :True ,          
                            
        "generator"    : Generator_Encoder_Net_CIFAR10,
        "discriminator": DiscriminatorNetCIFAR10, 
        "text_emb_model":Glove_Embbeding,
        "EmbeddingEncoder":None,
        
        "dataset": 'flowers-102',  #['flowers-102', 'mscoco', 'cifar-10', 'cifar-100',]
        "target_image_w":32,
        "target_image_h":32,
        "batchSize": 16,#128,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 4,
    }

    
   

    def __str__(self):
        return self.value


experimentsAll = [Experiment(experimentType=i) for i in ExperimentEnums]
