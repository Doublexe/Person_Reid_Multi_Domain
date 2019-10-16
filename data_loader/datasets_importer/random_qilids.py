from ..plain_datasets_importer.qilids import QiLIDS
import numpy as np
from .BaseDataset import BaseImageDataset

class Random_QiLIDS(BaseImageDataset):
    """A to B"""
    def __init__(self, cfg, verbose=True):
        data = QiLIDS(cfg.DATASETS.STORE_DIR, verbose=False).data

        query_ids = np.arange(119)
        np.random.shuffle(query_ids)
        query_ids = query_ids[:60]

        self.train = []
        self.query = []
        self.gallery = []
        counter = {idx: 0 for idx in query_ids}
        for ele in data:
            if ele[1] in query_ids and counter[ele[1]]<2:
                counter[ele[1]]+=1
                if counter[ele[1]] == 1:
                    self.query.append((ele[0], ele[1], 0))
                else:
                    self.gallery.append((ele[0], ele[1], 1))

        if verbose:
            print("=> Random QiLIDS Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
