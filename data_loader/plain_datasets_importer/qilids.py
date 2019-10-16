# encoding: utf-8

import os
import glob
import re
from .BaseDataset import BasePlainDataset

class QiLIDS(BasePlainDataset):

    """No Camera"""

    dataset_dir = 'QiLIDS'

    def __init__(self, store_dir, verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = os.path.join(store_dir, self.dataset_dir, 'images')

        self._check_before_run()

        data = self._process_dir(self.dataset_dir, relabel=True)

        if verbose:
            print("=> QiLIDS Loaded")
            self.print_dataset_statistics(data)

        self.data = data


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '**/*.jpg'), recursive=True)
        pattern = re.compile(r'([\d]{4})([\d]{3}).jpg')

        # Example: ./P1/cam2/238_0324.png

        dataset = []
        pid2label = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid2label.add(pid)
            dataset.append((img_path, pid, None))

        if relabel:
            temp = []
            pid2label = {pid: label for label, pid in enumerate(pid2label)}
            for data in dataset:
                temp.append((data[0], pid2label[data[1]], data[2]))
            dataset = temp
        return dataset
