from enum import Enum
from models.generators import GeneratorNetMSCOCO
from models.discriminators import  DiscriminatorNetMSCOCO
from torch import nn, optim
from experiment import Experiment


class ExperimentEnums(Enum):

    Mscoco = {
        "explainable": False,
        "explanationType": None,
        "generator": GeneratorNetMSCOCO,
        "discriminator": DiscriminatorNetMSCOCO,
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
