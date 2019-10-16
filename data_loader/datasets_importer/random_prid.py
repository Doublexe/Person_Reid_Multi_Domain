from .prid_ab import PRID_AB
import numpy as np
from .BaseDataset import BaseImageDataset

class Random_PRID(BaseImageDataset):
    """A to B"""
    def __init__(self, cfg, verbose=True):
        data = PRID_AB(cfg, verbose=False)
        query = data.query
        gallery = data.gallery

        query_ids = np.arange(200)
        np.random.shuffle(query_ids)
        query_ids = query_ids[:100]

        self.train = []
        self.query = []
        self.gallery = []
        for ele in query:
            if ele[1] in query_ids:
                self.query.append(ele)

        for ele in gallery:
            if ele[1] in query_ids or ele[1]>=200:
                self.gallery.append(ele)

        if verbose:
            print("=> Random PRID Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
