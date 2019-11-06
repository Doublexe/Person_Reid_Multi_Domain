import fire
import time
import torch
import numpy as np
import generalizers
import models
from data_loader import data_loader
from logger import make_logger
from evaluation import evaluation
from evaluation import re_ranking
from datasets import PersonReID_Dataset_Downloader
from utils import check_jupyter_run, get_last_stats
from utils.seed_utils import set_seeds
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def test_random_datasets(config_file, iteration=10, model_type="generalizer", **kwargs):

    if model_type == "normal":
        from config.default_multi_domain import _C as cfg
    elif model_type == "generalizer":
        from config.default_multi_domain import _C as cfg
    else:
        raise ValueError("Model type can only be normal or generalizer.")

    cfg.merge_from_file(config_file)
    if kwargs:
        opts = []
        for k,v in kwargs.items():
            opts.append(k)
            opts.append(v)
        cfg.merge_from_list(opts)
    cfg.freeze()

    # PersonReID_Dataset_Downloader('./datasets',cfg.DATASETS.NAMES)
    _, _, _, num_classes = data_loader(cfg,cfg.DATASETS.SOURCE, merge=cfg.DATASETS.MERGE)

    re_ranking=cfg.RE_RANKING

    device = torch.device(cfg.DEVICE)

    if model_type == "generalizer":
        module = getattr(generalizers, cfg.MODEL.NAME)
        model = getattr(module, 'Generalizer_G')(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.POOL)
        checkpoints = get_last_stats(cfg.OUTPUT_DIR)
        model_dict = torch.load(checkpoints[str(type(model))])
        model.load_state_dict(model_dict)

    elif model_type == "normal":
        model = getattr(models, cfg.MODEL.NAME)(
            num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.POOL)
        checkpoints = get_last_stats(cfg.OUTPUT_DIR, [cfg.MODEL.NAME])
        model_dict = torch.load(checkpoints[cfg.MODEL.NAME])
        model.load_state_dict(model_dict)

    model = model.eval()

    if not re_ranking:
        logger = make_logger("Reid_Baseline", cfg.OUTPUT_DIR,
                             'epo'+str(checkpoints['epo']))
        logger.info("Test Results:")
    else:
        logger = make_logger("Reid_Baseline", cfg.OUTPUT_DIR,
                             'epo'+str(checkpoints['epo'])+'_re-ranking')
        logger.info("Re-Ranking Test Results:")


    for test_dataset in cfg.DATASETS.TARGET:
        mAPs = []
        cmcs = []
        for i in range(iteration):

            set_seeds(i)

            _, val_loader, num_query, _ = data_loader(cfg,(test_dataset,), merge=False)

            all_feats = []
            all_pids = []
            all_camids = []

            since = time.time()
            for data in tqdm(val_loader, desc='Feature Extraction', leave=False):
                model.eval()
                with torch.no_grad():
                    images, pids, camids = data
                    if device:
                        model.to(device)
                        images = images.to(device)

                    feats = model(images)
                    feats /= feats.norm(dim=-1, keepdim=True)

                all_feats.append(feats)
                all_pids.extend(np.asarray(pids))
                all_camids.extend(np.asarray(camids))

            cmc, mAP = evaluation(all_feats,all_pids,all_camids,num_query,re_ranking)
            mAPs.append(mAP)
            cmcs.append(cmc)


        mAP = np.mean(np.array(mAPs))
        cmc = np.mean(np.array(cmcs), axis=0)

        mAP_std = np.std(np.array(mAPs))
        cmc_std = np.std(np.array(cmcs), axis=0)

        logger.info("mAP: {:.1%} (std: {:.3%})".format(mAP, mAP_std))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%} (std: {:.3%})".format(r, cmc[r - 1], cmc_std[r - 1]))

    test_time = time.time() - since
    logger.info('Testing complete in {:.0f}m {:.0f}s'.format(test_time // 60, test_time % 60))


if __name__=='__main__':
    fire.Fire(test_random_datasets)
