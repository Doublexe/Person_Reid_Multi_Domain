# encoding: utf-8

import os
import glob
import re
from .BaseDataset import BasePlainDataset

class ILIDS(BasePlainDataset):

    dataset_dir = 'iLIDS'

    def __init__(self, store_dir, verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = os.path.join(store_dir, self.dataset_dir, 'i-LIDS-VID', 'images')

        self._check_before_run()

        data = self._process_dir(self.dataset_dir, relabel=True)

        if verbose:
            print("=> iLIDS Loaded")
            self.print_dataset_statistics(data)

        self.data = data


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '**/*.png'), recursive=True)
        pattern = re.compile(r'cam([\d])_person([\d]+).png')

        # Example: ./P1/cam2/238_0324.png

        dataset = []
        pid2label = set()
        for img_path in img_paths:
            camid, pid = map(int, pattern.search(img_path).groups())
            pid -= 1
            camid -= 1
            pid2label.add(pid)
            dataset.append((img_path, pid, camid))

        if relabel:
            temp = []
            pid2label = {pid: label for label, pid in enumerate(pid2label)}
            for data in dataset:
                temp.append((data[0], pid2label[data[1]], data[2]))
            dataset = temp
        return dataset
