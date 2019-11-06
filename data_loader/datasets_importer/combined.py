# encoding: utf-8
from collections import OrderedDict
import numpy as np


class Combined:
    def get_imagedata_info(self, data):
        pids, cams, domains = [], [], []
        for _, pid, camid, domain in data:
            pids += [pid]
            cams += [camid]
            domains += [domain]
        pids = set(pids)
        cams = set(cams)
        domains = set(domains)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_domains = len(domains)
        return num_pids, num_imgs, num_cams, num_domains


    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_domains = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_query_domains = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_domains = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ---------------------------------------------------")
        print("  subset   | # ids | # images | # cameras | # domains")
        print("  ---------------------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams, num_train_domains))
        print("  query    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams, num_query_domains))
        print("  gallery  | {:5d} | {:8d} | {:9d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_domains))
        print("  ---------------------------------------------------")


    def __init__(self, datasets, merge=False, verbose=True, **kwargs):
        """
        Parameter
        ---------
        datasets : Dict{name -> BaseImageDataset}
            A dictionary of datasets to combine.
        merge : bool
            If merge, merge all train, query, gallery into train, and query&gallery will be empty. See 'Return'.

        Return
        ---------
        self.train, self.query, self.gallery : List[img_path, pid, camid, domain]
        self.order : List[str]
            A list of dataset names. The ordering of all other parameters follow this order.
        self.*_pid_offset : List[int]
            The offsets for pid for different datasets.
        self.*_camid_offset : List[int]
            The offsets for camid for different datasets.

        Assume:
            1. No negetive ids;
            2. The new arragement of ids will be (for both camid and pid):
                [ D1:train D2:train ... D1:others D2:others ... ]
            3. Train camids do not intersect with others' camids
        """
        super(Combined, self).__init__()
        self.datasets = OrderedDict(datasets.items())
        self.order = [key for key in self.datasets.keys()]

        train_pid_range = [dataset.num_train_pids for _, dataset in self.datasets.items()]
        train_camid_max = [np.asarray(dataset.train)[:,2].astype(int).max(0)+1 if len(dataset.train)!=0 else 0 for _, dataset in self.datasets.items()]
        others_stat = [np.asarray(dataset.query+dataset.gallery)[:,1:].astype(int).max(0)+1 if len(dataset.query+dataset.gallery)!=0 else (0,0) for _, dataset in self.datasets.items()]
        others_pid_max = [stat[0] for stat in others_stat]
        others_camid_max = [stat[1] for stat in others_stat]

        self.train_pid_offset = [sum(train_pid_range[:self.order.index(name)]) for name in self.order]
        self.train_camid_offset = [sum(train_camid_max[:self.order.index(name)]) for name in self.order]
        upper_train_pid = sum(train_pid_range)
        upper_train_camid = sum(train_camid_max)
        self.others_pid_offset = [upper_train_pid+sum(others_pid_max[:self.order.index(name)]) for name in self.order]
        self.others_camid_offset = [upper_train_camid+sum(others_camid_max[:self.order.index(name)]) for name in self.order]

        train = [(t[0], t[1]+self.train_pid_offset[self.order.index(name)], t[2]+self.train_camid_offset[self.order.index(name)], self.order.index(name))
                    for name, dataset in self.datasets.items() for t in dataset.train]
        query = [(t[0], t[1]+self.others_pid_offset[self.order.index(name)], t[2]+self.others_camid_offset[self.order.index(name)], self.order.index(name))
                    for name, dataset in self.datasets.items() for t in dataset.query]
        gallery = [(t[0], t[1]+self.others_pid_offset[self.order.index(name)], t[2]+self.others_camid_offset[self.order.index(name)], self.order.index(name))
                    for name, dataset in self.datasets.items() for t in dataset.gallery]

        if merge:
            pid_container = set()
            self.others_pid_offset = np.array(self.others_pid_offset)
            train = query+gallery+train
            gallery = []
            query = []
            for t in train:
                pid_container.add(t[1])
            pids = [pid for pid in pid_container]
            pids.sort()
            pid2label={}
            for label, pid in enumerate(pids):
                pid2label[pid] = label
                if (pid+1) in self.others_pid_offset:
                    self.others_pid_offset[self.others_pid_offset==(pid+1)] = label+1
            train = [(t[0], pid2label[t[1]], t[2], t[3]) for t in train]
            self.others_pid_offset=self.others_pid_offset.tolist()


        if verbose:
            print("=> Combined{}: {} Loaded".format(' Merged' if merge else '',','.join(self.order)))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_domains = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_domains = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_domains = self.get_imagedata_info(self.gallery)
