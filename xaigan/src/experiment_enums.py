from enum import Enum

from models.generators import EmbeddingEncoderNetMSCOCO
from models.generators import GeneratorNet_MSCOCO,GeneratorNet_2_MSCOCO,Generator_Encoder_Net_MSCOCO,GeneratorNet_2_EncoderNet_MSCOCO
from models.generators import GeneratorNetCIFAR10,Generator_Encoder_Net_CIFAR10
from models.discriminators import DiscriminatorNetCIFAR10,DiscriminatorNetMSCOCO,DiscriminatorNetMSCOCO_2
from models.text_embedding_models import RobertaClass,Glove_Embbeding

from torch import nn, optim
from experiment import Experiment


class ExperimentEnums(Enum):
    #
    # flowers_only = {
    #     "explainable"     :True,
    #     "explanationType" :"saliency",
    #     "noise_emb_sz"    :100,            #GeneratorNetMSCOCO's noise param
    #     "text_emb_sz"     :768,    #TODO:  #RobertaClass's param
    #     'Encoder_emb_sz'  :100,
    #     "text_max_len"    :350,            #RobertaClass's param
    #     "use_CLS_emb"     :False,          #RobertaClass's param
    #     "use_one_caption" :False ,         #RobertaClass's param + param used in experiment
    #     "use_captions"    :False ,
    #     "use_captions_only"    :False ,
    #
    #     "generator"    : GeneratorNetCIFAR10,
    #     "discriminator": DiscriminatorNetCIFAR10,
    #     "text_emb_model"  :None,
    #     "EmbeddingEncoder":None,
    #
    #     "dataset": 'flowers-102',  #['flowers-102', 'mscoco', 'cifar-10', 'cifar-100',]
    #     "target_image_w":32,
    #     "target_image_h":32,
    #     "batchSize": 16,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 100
    # }
    
    # flowers_Roberta = {
    #     "explainable"     :False,
    #     "explanationType" :None,
    #     "noise_emb_sz"    :100,            #GeneratorNetMSCOCO's noise param
    #     "text_emb_sz"     :768,    #TODO:  #RobertaClass's param
    #     'Encoder_emb_sz'  :(100+767)//2,
    #     "text_max_len"    :350,            #RobertaClass's param
    #     "use_CLS_emb"     :False,          #RobertaClass's param
    #     "use_one_caption" :True ,          #RobertaClass's param + param used in experiment 
    #     "use_captions"    :True ,          
    #     "use_captions_only"    :False ,
                            
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
    
    # flowers_Glove_2 = {
    #     "explainable"     :False,
    #     "explanationType" :None,
    #     "noise_emb_sz"    :100,            #GeneratorNetMSCOCO's noise param
    #     "text_emb_sz"     :50,         #Glove param word emb-> sent emb
    #     'Encoder_emb_sz'  :100+50,
    #     "text_max_len"    :350,            #Glove param
    #     "use_CLS_emb"     :False,          #Glove param
    #     "use_one_caption" :True ,          #Glove param + param used in experiment 
    #     "use_captions"    :True ,          
    #     "use_captions_only"    :False,
                            
    #     "generator"    : Generator_Encoder_Net_CIFAR10,
    #     "discriminator": DiscriminatorNetCIFAR10, 
    #     "text_emb_model":Glove_Embbeding,
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
    #     "epochs": 4,
    # }

    flowers_Roberta_only = {
        "explainable"     :False,
        "explanationType" :None,
        "noise_emb_sz"    :100,            #GeneratorNetMSCOCO's noise param
        "text_emb_sz"     :768,    #TODO:  #RobertaClass's param
        'Encoder_emb_sz'  :(100+767)//2,
        "text_max_len"    :350,            #RobertaClass's param
        "use_CLS_emb"     :False,          #RobertaClass's param
        "use_one_caption" :True ,          #RobertaClass's param + param used in experiment 
        "use_captions"    :True ,          
        "use_captions_only"    :True ,
                            
        "generator"    : GeneratorNetCIFAR10,
        "discriminator": DiscriminatorNetCIFAR10, 
        "text_emb_model":RobertaClass,
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
        "epochs": 30
    }

    # flowers_Glove_2_only = {
    #     "explainable"     :False,
    #     "explanationType" :None,
    #     "noise_emb_sz"    :100,            #GeneratorNetMSCOCO's noise param
    #     "text_emb_sz"     :50,             #Glove param word emb-> sent emb
    #     'Encoder_emb_sz'  :100+50,
    #     "text_max_len"    :350,            #Glove param
    #     "use_CLS_emb"     :False,          #Glove param
    #     "use_one_caption" :True ,          #Glove param + param used in experiment 
    #     "use_captions"    :True ,          
    #     "use_captions_only"    :True ,
        
                            
    #     "generator"    : GeneratorNetCIFAR10,
    #     "discriminator": DiscriminatorNetCIFAR10, 
    #     "text_emb_model":Glove_Embbeding,
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
    #     "epochs": 10,
    # }


######
    # CIFAR10_only_SaliencyTrain = {
    #     "explainable"     :True,
    #     "explanationType" :"saliency",
    #     "noise_emb_sz"    :100,            #GeneratorNetMSCOCO's noise param
    #     "text_emb_sz"     :768,    #TODO:  #RobertaClass's param
    #     'Encoder_emb_sz'  :(100+767)//2,
    #     "text_max_len"    :350,            #RobertaClass's param
    #     "use_CLS_emb"     :False,          #RobertaClass's param
    #     "use_one_caption" :False ,         #RobertaClass's param + param used in experiment
    #     "use_captions"    :False ,
    #     "use_captions_only"    :False ,


    #     "generator"    : GeneratorNetCIFAR10,
    #     "discriminator": DiscriminatorNetCIFAR10,
    #     "text_emb_model":None,
    #     "EmbeddingEncoder":None,

    #     "dataset": 'cifar-10',  #['flowers-102', 'mscoco', 'cifar-10', 'cifar-100',]
    #     "target_image_w":32,
    #     "target_image_h":32,
    #     "batchSize": 16,
    #     "percentage": 1,
    #     "g_optim": optim.Adam,
    #     "d_optim": optim.Adam,
    #     "glr": 0.0002,
    #     "dlr": 0.0002,
    #     "loss": nn.BCELoss(),
    #     "epochs": 200
    # }
  
   

    def __str__(self):
        return self.value


experimentsAll = [Experiment(experimentType=i) for i in ExperimentEnums]
