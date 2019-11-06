import fire
import os
import time
import torch
import numpy as np
import models
from config import cfg
from data_loader import data_loader
from loss import make_loss
from optimizer import make_optimizer
from scheduler import make_scheduler
from logger import make_logger
from evaluation import evaluation
import random
from datasets import PersonReID_Dataset_Downloader
from utils.get_last_stats import get_last_stats
from utils.seed_utils import reset, set_seeds
from utils import check_jupyter_run
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

iteration=10

def train(config_file, resume=False, **kwargs):
    cfg.merge_from_file(config_file)
    if kwargs:
        opts = []
        for k,v in kwargs.items():
            opts.append(k)
            opts.append(v)
        cfg.merge_from_list(opts)
    cfg.freeze()

    # [PersonReID_Dataset_Downloader(cfg.DATASETS.STORE_DIR,dataset) for dataset in cfg.DATASETS.SOURCE]
    # [PersonReID_Dataset_Downloader(cfg.DATASETS.STORE_DIR,dataset) for dataset in cfg.DATASETS.TARGET]
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = make_logger("Reid_Baseline", output_dir,'log', resume)
    if not resume:
        logger.info("Using {} GPUS".format(1))
        logger.info("Loaded configuration file {}".format(config_file))
        logger.info("Running with config:\n{}".format(cfg))

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = torch.device(cfg.DEVICE)
    epochs = cfg.SOLVER.MAX_EPOCHS

    train_loader, _, _, num_classes = data_loader(cfg,cfg.DATASETS.SOURCE, merge=cfg.DATASETS.MERGE)

    model = getattr(models, cfg.MODEL.NAME)(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.POOL)
    if resume:
        checkpoints = get_last_stats(output_dir)
        try:
            model_dict = torch.load(checkpoints[cfg.MODEL.NAME])
        except KeyError:
            model_dict = torch.load(checkpoints[str(type(model))])
        model.load_state_dict(model_dict)
        if device:
            model.to(device)  # must be done before the optimizer generation
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg,optimizer)
    base_epo = 0
    if resume:
        optimizer.load_state_dict(torch.load(checkpoints['opt']))
        sch_dict = torch.load(checkpoints['sch'])
        scheduler.load_state_dict(sch_dict)
        base_epo = checkpoints['epo']

    loss_fn = make_loss(cfg)

    if not resume:
        logger.info("Start training")
    since = time.time()
    for epoch in range(epochs):
        count = 0
        running_loss = 0.0
        running_acc = 0
        for data in tqdm(train_loader, desc='Iteration', leave=False):
            model.train()
            images, labels, domains = data
            if device:
                model.to(device)
                images, labels, domains = images.to(device), labels.to(device), domains.to(device)

            optimizer.zero_grad()

            scores, feats = model(images)
            loss = loss_fn(scores, feats, labels)

            loss.backward()
            optimizer.step()

            count = count + 1
            running_loss += loss.item()
            running_acc += (scores[0].max(1)[1] == labels).float().mean().item()


        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch+1+base_epo, count, len(train_loader),
                                    running_loss/count, running_acc/count,
                                    scheduler.get_lr()[0]))
        scheduler.step()

        if (epoch+1+base_epo) % checkpoint_period == 0:
            model.cpu()
            model.save(output_dir,epoch+1+base_epo)
            torch.save(optimizer.state_dict(), os.path.join(output_dir, 'opt_epo'+str(epoch+1+base_epo)+'.pth'))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, 'sch_epo'+str(epoch+1+base_epo)+'.pth'))


        # Validation
        if (epoch+base_epo + 1) % eval_period == 0:
            # Validation on Target Dataset
            for target in cfg.DATASETS.TARGET:
                mAPs = []
                cmcs = []
                for i in range(iteration):

                    set_seeds(i)

                    _, val_loader, num_query, _ = data_loader(cfg,(target,), merge=False)

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

                    cmc, mAP = evaluation(all_feats,all_pids,all_camids,num_query)
                    mAPs.append(mAP)
                    cmcs.append(cmc)


                mAP = np.mean(np.array(mAPs))
                cmc = np.mean(np.array(cmcs), axis=0)

                mAP_std = np.std(np.array(mAPs))
                cmc_std = np.std(np.array(cmcs), axis=0)

                logger.info("Validation Results: {} - Epoch: {}".format(target, epoch+1+base_epo))
                logger.info("mAP: {:.1%} (std: {:.3%})".format(mAP, mAP_std))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%} (std: {:.3%})".format(r, cmc[r - 1], cmc_std[r - 1]))

            reset()



    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('-' * 10)


if __name__=='__main__':
    import fire
    fire.Fire(train)
