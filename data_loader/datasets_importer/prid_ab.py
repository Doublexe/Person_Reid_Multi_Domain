# encoding: utf-8

import os
import glob
import re
from .BaseDataset import BaseImageDataset


class PRID_AB(BaseImageDataset):
    """
    A -> B

    # WARNING: single shot instead of multiple shot.
                pid >= 200 for unrelated gallery images.
    """

    dataset_dir = 'PRID'

    def __init__(self, cfg, verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = os.path.join(cfg.DATASETS.STORE_DIR, self.dataset_dir, 'single_shot')

        self.query_dir = os.path.join(self.dataset_dir, 'cam_a')
        self.gallery_dir = os.path.join(self.dataset_dir, 'cam_b')

        self._check_before_run()

        train = []
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)

        if verbose:
            print("=> PRID A->B Loaded")
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
        domain = os.path.basename(dir_path)
        img_paths = glob.glob(os.path.join(dir_path, '*.png'))
        pattern = re.compile(r'^person_([-\d]+).png')

        dataset = []
        for img_path in img_paths:
            pid = list(map(int, pattern.search(os.path.basename(img_path)).groups()))[0]
            camid = 0 if domain=='cam_a' else 1  # index starts from 0
            if domain=='cam_a':
                if pid <= 200:
                    pid -=1
                    dataset.append((img_path, pid, camid))
            else:
                pid -=1
                dataset.append((img_path, pid, camid))

        return dataset
