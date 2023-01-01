from enum import Enum
from models.generators import GeneratorNetMSCOCO,GeneratorNet_TEXT2IMG_MSCOCO
from models.discriminators import  DiscriminatorNetMSCOCO,DiscriminatorNet_TEXT2IMG_MSCOCO
from models.text_embedding_models import RobertaClass
from models.encoders import EmbeddingEncoderNetMSCOCO

from torch import nn, optim
from experiment import Experiment


class ExperimentEnums(Enum):

    Mscoco = {
        "explainable": False,
        "explanationType": None,
        "noise_emb_sz" : 100,       #GeneratorNetMSCOCO's noise param
        "text_emb_sz" : 768, #TODO: #RobertaClass's param
        "text_max_len":350,         #RobertaClass's param
        "use_one_caption": True,    #RobertaClass's param
        "use_CLS_emb":False,         #RobertaClass's param

        # "generator": GeneratorNetMSCOCO,
        # "discriminator": DiscriminatorNetMSCOCO,
        "generator": GeneratorNet_TEXT2IMG_MSCOCO,
        "discriminator": DiscriminatorNet_TEXT2IMG_MSCOCO,

        "text_emb_model":RobertaClass,
        "EmbeddingEncoder":EmbeddingEncoderNetMSCOCO,
        "dataset": "mscoco",
        "batchSize": 100,#128,
        "percentage": 1,
        "g_optim": optim.Adam,
        "d_optim": optim.Adam,
        "glr": 0.0002,
        "dlr": 0.0002,
        "loss": nn.BCELoss(),
        "epochs": 50
    }
   

    def __str__(self):
        return self.value


experimentsAll = [Experiment(experimentType=i) for i in ExperimentEnums]
