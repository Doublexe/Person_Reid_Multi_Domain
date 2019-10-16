# encoding: utf-8
from .cuhk03 import CUHK03
from .market1501 import Market1501
from .dukemtmc import DukeMTMC
from .msmt17 import MSMT17
#from .ntucampus import NTUCampus
from .ntuindoor import NTUIndoor
from .ImageDataset import ImageDataset
from .random_viper import Random_VIPeR
from .random_grid import Random_GRID
from .random_prid import Random_PRID
from .random_qilids import Random_QiLIDS
from .cuhk02 import CUHK02
from .cuhk_sysu import CUHK_SYSU
from .combined import Combined
from collections import OrderedDict
# from .

__factory = {
    'CUHK03': CUHK03,
    'Market1501': Market1501,
    'DukeMTMC': DukeMTMC,
    'MSMT17': MSMT17,
    'NTUIndoor': NTUIndoor,
    'Random_VIPeR': Random_VIPeR,
    'Random_GRID': Random_GRID,
    'Random_PRID': Random_PRID,
    'Random_QiLIDS': Random_QiLIDS,
    'CUHK02': CUHK02,
    'CUHK-SYSU': CUHK_SYSU,

}


def get_names():
    return __factory.keys()


def init_dataset(cfg,dataset_names, merge, *args, **kwargs):
    for dataset_name in dataset_names:
        if dataset_name not in __factory.keys():
            raise KeyError("Unknown datasets: {}".format(dataset_name))
    if len(dataset_names) == 1:
        return __factory[dataset_names[0]](cfg,*args, **kwargs)
    else:
        datasets = OrderedDict([(name, __factory[name](cfg,*args, **kwargs)) for name in dataset_names])
        return Combined(datasets, merge)
