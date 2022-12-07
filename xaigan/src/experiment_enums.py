from enum import Enum
from models.generators import GeneratorNetMSCOCO
from models.discriminators import  DiscriminatorNetMSCOCO
from models.text_embedding_models import RobertaClass

from torch import nn, optim
from experiment import Experiment


class ExperimentEnums(Enum):

    Mscoco = {
        "explainable": False,
        "explanationType": None,
        "generator": GeneratorNetMSCOCO,
        "discriminator": DiscriminatorNetMSCOCO,
        "text_emb_model":RobertaClass,
        "text_max_len":350,       #RobertaClass's param
        "use_one_caption": True,  #RobertaClass's param
        "use_CLS_emb":True,       #RobertaClass's param
        # "text_emb_size": 768,   #TODO: RobertaClass's param
        "dataset": "mscoco",
        "batchSize": 128,
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
