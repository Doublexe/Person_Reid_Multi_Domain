from ..plain_datasets_importer.cuhk02 import CUHK02 as HK02
from .BaseDataset import BaseImageDataset

class CUHK02(BaseImageDataset):
    def __init__(self, cfg, verbose=True):
        data = HK02(cfg.DATASETS.STORE_DIR, verbose=False).data

        self.train = data
        self.query = []
        self.gallery = []


        if verbose:
            print("=> CUHK02 Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
