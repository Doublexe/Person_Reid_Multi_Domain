# encoding: utf-8

import os
import glob
import re
from .BaseDataset import BaseImageDataset


class GRID(BaseImageDataset):
    """
    8 cameras?

    # WARNING: The pid 0 represents multiple identities (unrelated gallery images).
    """

    dataset_dir = 'GRID'

    def __init__(self, cfg, verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = os.path.join(cfg.DATASETS.STORE_DIR, self.dataset_dir)

        self.query_dir = os.path.join(self.dataset_dir, 'probe')
        self.gallery_dir = os.path.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = []
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)

        if verbose:
            print("=> GRID Loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpeg'))
        pattern = re.compile(r'^([-\d]+)_([-\d]+)')

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(os.path.basename(img_path)).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if pid == 0:
                dataset.append((img_path, 0, camid))
            else:
                dataset.append((img_path, pid, camid))

        return dataset
