from ..plain_datasets_importer.cuhk_sysu import CUHK_SYSU as SYSU
from .BaseDataset import BaseImageDataset

class CUHK_SYSU(BaseImageDataset):
    def __init__(self, cfg, verbose=True):
        data = SYSU(cfg.DATASETS.STORE_DIR, verbose=False).data

        self.train = data
        self.query = []
        self.gallery = []


        if verbose:
            print("=> CUHK-SYSU Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
