from .grid import GRID
import numpy as np
from .BaseDataset import BaseImageDataset

class Random_GRID(BaseImageDataset):
    def __init__(self, cfg, verbose=True):
        data = GRID(cfg, verbose=False)
        query = data.query
        gallery = data.gallery

        query_ids = np.arange(250) + 1
        np.random.shuffle(query_ids)
        query_ids = query_ids[:125]

        self.train = []
        self.query = []
        self.gallery = []
        for ele in query:
            if ele[1] in query_ids:
                self.query.append(ele)

        for ele in gallery:
            if ele[1] in query_ids or ele[1]==0:
                self.gallery.append(ele)

        if verbose:
            print("=> Random GRID Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
