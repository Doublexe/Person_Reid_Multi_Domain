from ..plain_datasets_importer.viper import VIPeR
import numpy as np
from .BaseDataset import BaseImageDataset

class Random_VIPeR(BaseImageDataset):
    def __init__(self, cfg, verbose=True):
        data = VIPeR(cfg.DATASETS.STORE_DIR, verbose=False).data
        query_ids = np.arange(632)
        np.random.shuffle(query_ids)
        query_ids = query_ids[:316]

        self.train = []
        self.query = []
        self.gallery = []
        for ele in data:
            if ele[1] in query_ids:
                if ele[2]==0:
                    self.query.append(ele)
                elif ele[2]==1:
                    self.gallery.append(ele)

        if verbose:
            print("=> Random VIPeR Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
