import os
import numpy as np
import models
from config import cfg
from data_loader import data_loader
from datasets import PersonReID_Dataset_Downloader
from matplotlib import pyplot as plt
from utils import check_jupyter_run
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm



class Explorer(object):
    def __init__(self, config_file, epoch_label, **kwargs):
        """
        Validation set is split into two parts - query (probe) and gallery (to be searched), based on num_query.

        ::Return: Initialize a file 'model_epoch.mtch':
                matching matrix M of num_query x num_gallery. M_ij is 1 <=> ith query is matched at rank j.
        """

        cfg.merge_from_file(config_file)

        if kwargs:
            opts = []
            for k,v in kwargs.items():
                opts.append(k)
                opts.append(v)
            cfg.merge_from_list(opts)
        cfg.freeze()
        self.cfg = cfg

        device = torch.device(cfg.DEVICE)
        output_dir = cfg.OUTPUT_DIR
        epoch = epoch_label
        re_ranking = cfg.RE_RANKING
        if not os.path.exists(output_dir):
            raise OSError('Output directory does not exist.')
        save_filename = (cfg.MODEL.NAME + '_epo%s.mtch' % epoch_label)
        self._filepath = os.path.join(output_dir,save_filename)

        if os.path.exists(self._filepath):
            print('Loading matches file...')
            self.data = np.load(self._filepath)
            train_loader, val_loader, num_query, num_classes = data_loader(cfg,cfg.DATASETS.NAMES)
            self.dataset = val_loader.dataset
            print('Matches loaded.')
        else:
            print('Creating matches file...')
            PersonReID_Dataset_Downloader(cfg.DATASETS.STORE_DIR, cfg.DATASETS.NAMES)

            train_loader, val_loader, num_query, num_classes = data_loader(cfg,cfg.DATASETS.NAMES)

            # load model
            model = getattr(models, cfg.MODEL.NAME)(num_classes)
            model.load(output_dir, epoch)
            model.eval()

            all_feats = []
            all_pids = []
            all_camids = []
            for data in tqdm(val_loader, desc='Feature Extraction', leave=False):
                with torch.no_grad():

                    images, pids, camids = data

                    if device:
                        model.to(device)
                        images = images.to(device)

                    feats = model(images)

                all_feats.append(feats)
                all_pids.extend(np.asarray(pids))
                all_camids.extend(np.asarray(camids))

            all_feats = torch.cat(all_feats, dim=0)
            # query
            qf = all_feats[:num_query]
            q_pids = np.asarray(all_pids[:num_query])
            q_camids = np.asarray(all_camids[:num_query])

            # gallery
            gf = all_feats[num_query:]
            g_pids = np.asarray(all_pids[num_query:])
            g_camids = np.asarray(all_camids[num_query:])

            if re_ranking:
                raise NotImplementedError()
            else:
                m, n = qf.shape[0], gf.shape[0]
                distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(1, -2, qf, gf.t())
                distmat = distmat.cpu().numpy()

            indices = np.argsort(distmat, axis=1)
            # matches = np.repeat(g_pids.reshape([1, n]), m, axis=0) == q_pids[:, np.newaxis]
            ranked_matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

            data = {
                'q_pids': q_pids,
                'g_pids': g_pids,
                'q_camids': q_camids,
                'g_camids': g_camids,
                'ranked_matches': ranked_matches,
                # 'matches': matches,
                'indices': indices,
            }

            # save as .mtch
            with open(self._filepath, 'wb') as f:
                np.savez(f, **data)

            print('Matches created.')

            self.data = data
            self.dataset = val_loader.dataset

    def _recover_image(self, image):
        # ::Return: denormalize the image and perumte dim to right order.
        #           Final format is still Tensor.
        image = image.permute(1, 2, 0) * torch.Tensor(self.cfg.INPUT.PIXEL_STD) + torch.Tensor(self.cfg.INPUT.PIXEL_MEAN)
        return image

    def _get_dataset_idx(self, q_idx, g_idx):
        # Given q_idx, g_idx in matching matrix, return the index in dataset
        return len(self.data['q_pids']) + g_idx

    def show_by_qeury_pid(self, q_pid, option='all', limit=10):
        # Show images: rows, each row containing images related to the query
        #
        # Option:
        #   1. all: by ranking, all images
        #   2. mistake: by ranking, all mistakes
        #
        # Limit: maxmium number of images to show
        idxs = [idx for idx, ele in enumerate(self.data['q_pids']) if ele == q_pid]

        fig, axes = plt.subplots(nrows=1, ncols=len(idxs))
        for num, idx in enumerate(idxs):
            self._show_by_dataset_idx(idx, match=True, ax=axes[num], rank=0)

        for idx in idxs:
            self.show_by_query_index(idx, option=option, limit=limit)

    def show_by_query_index(self, q_idx,option='all',limit=10):
        if option == 'all':
            fig, axes = plt.subplots(nrows=1, ncols=limit)
            for rank in range(limit):
                g_idx = self.data['indices'][q_idx, rank]
                match = self.data['ranked_matches'][q_idx, rank]
                d_idx = self._get_dataset_idx(q_idx, g_idx)
                self._show_by_dataset_idx(d_idx, match, axes[rank], rank)
            plt.tight_layout()
            plt.show()
        elif option == 'mistake':
            fig, axes = plt.subplots(nrows=1, ncols=limit)
            count=0
            for rank in range(limit):
                g_idx = self.data['indices'][q_idx, rank]
                match = self.data['ranked_matches'][q_idx, rank]
                d_idx = self._get_dataset_idx(q_idx, g_idx)
                if match == 0:
                    self._show_by_dataset_idx(d_idx, match, axes[rank], count)
                    count +=1

            plt.tight_layout()
            plt.show()
        else:
            raise KeyError('Option not valid. It should be: all / mistake.')

    def save_by_query_index(self, path, q_idx, limit=20):
        image, _, _, _ = self.dataset[q_idx]
        image = self._recover_image(image)
        plt.imsave(os.path.join(path,'query'+'.png'), image)
        for rank in range(limit):
            g_idx = self.data['indices'][q_idx, rank]
            match = self.data['ranked_matches'][q_idx, rank]
            d_idx = self._get_dataset_idx(q_idx, g_idx)
            image, _, _, _ = self.dataset[d_idx]
            image = self._recover_image(image)
            if match:
                plt.imsave(os.path.join(path,'r'+str(rank)+'_true'+'.png'), image)
            else:
                plt.imsave(os.path.join(path,'r'+str(rank)+'_false'+'.png'), image)

    def _show_by_dataset_idx(self, d_idx, match, ax, rank):
        image, pid, camid, path = self.dataset[d_idx]
        image = self._recover_image(image)
        ax.imshow(image)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title('P'+str(pid)+'C'+str(camid)+'R'+str(rank), fontsize=6)
        if match:
            ax.spines['bottom'].set_color('green')
            ax.spines['top'].set_color('green')
            ax.spines['right'].set_color('green')
            ax.spines['left'].set_color('green')
        else:
            ax.spines['bottom'].set_color('red')
            ax.spines['top'].set_color('red')
            ax.spines['right'].set_color('red')
            ax.spines['left'].set_color('red')



if __name__ == "__main__":
    explore = Explorer('./config/market_softmax_triplet.yaml', 120)
#%%
