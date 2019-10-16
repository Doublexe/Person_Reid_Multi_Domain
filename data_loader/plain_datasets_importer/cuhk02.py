# encoding: utf-8

import os
import glob
import re
from .BaseDataset import BasePlainDataset
import pickle

class CUHK02(BasePlainDataset):

    dataset_dir = 'CUHK02'
    cam_pairs = ['P1', 'P2', 'P3', 'P4', 'P5']

    def __init__(self, store_dir, verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = os.path.join(store_dir, self.dataset_dir)

        self._check_before_run()

        dataset_pth = os.path.join(self.dataset_dir, 'dataset.list')
        if os.path.exists(dataset_pth):
            with open(dataset_pth, 'rb') as f:
                data = pickle.load(f)
        else:
            data = self._process_dir(self.dataset_dir, relabel=True)
            with open(dataset_pth, 'wb') as f:
                pickle.dump(data, f)

        if verbose:
            print("=> CUHK02 Loaded")
            self.print_dataset_statistics(data)
        self.data = data


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '**/*.png'), recursive=True)
        pattern = re.compile(r'P([\d])\/cam([\d])\/([\d]+)_([\d]+)')

        # Example: ./P1/cam2/238_0324.png

        pid2label_pth = os.path.join(self.dataset_dir, 'pid2label.dict')
        if os.path.exists(pid2label_pth):
            with open(pid2label_pth, 'rb') as f:
                pid2label = pickle.load(f)
        else:
            pid_container = set()
            for img_path in img_paths:
                P, cam, pid, _ = map(int, pattern.search(img_path).groups())
                if pid == -1: continue  # junk images are just ignored
                pid_container.add('_'.join([str(P), str(pid)]))
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            with open(pid2label_pth, 'wb') as f:
                pickle.dump(pid2label, f)

        dataset = []
        for img_path in img_paths:
            P, cam, pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid = (P-1)*2 + cam-1  # index starts from 0
            pid = '_'.join([str(P), str(pid)])
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
