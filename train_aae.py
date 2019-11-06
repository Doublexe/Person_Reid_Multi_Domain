import generalizers
import fire
import os
import time
import torch
from torch import nn
import numpy as np
from config.default_multi_domain import _C as cfg
from data_loader import data_loader
from torch.optim import Adam
from scheduler import make_scheduler
from logger import make_logger
from evaluation import evaluation
from datasets import PersonReID_Dataset_Downloader
from utils import check_jupyter_run
from utils.seed_utils import set_seeds, reset
from loss.softlabel_loss import LabelSmoothingLoss
# from torch.utils.tensorboard import SummaryWriter
from utils.get_last_stats import get_last_stats
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

EPS = 1e-12

def train(config_file, resume=False, iteration = 10, STEP=4, **kwargs):
    """
    Parameter
    ---------
    resume : bool
        If true, continue the training and append logs to the previous log.
    iteration : int
        number of loops to test Random Datasets.
    STEP : int
        Number of steps to train the discriminator per batch
    """


    cfg.merge_from_file(config_file)
    if kwargs:
        opts = []
        for k, v in kwargs.items():
            opts.append(k)
            opts.append(v)
        cfg.merge_from_list(opts)
    cfg.freeze()

    # [PersonReID_Dataset_Downloader('./datasets', name) for name in cfg.DATASETS.NAMES]

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
    sources = cfg.DATASETS.SOURCE
    target = cfg.DATASETS.TARGET
    pooling = cfg.MODEL.POOL
    last_stride = cfg.MODEL.LAST_STRIDE


    # tf_board_path = os.path.join(output_dir, 'tf_runs')
    # if os.path.exists(tf_board_path):
    #     shutil.rmtree(tf_board_path)
    # writer = SummaryWriter(tf_board_path)


    gan_d_param=cfg.MODEL.D_PARAM
    gan_g_param=cfg.MODEL.G_PARAM
    class_param=cfg.MODEL.CLASS_PARAM

    """Set up"""
    train_loader, _, _, num_classes = data_loader(cfg,cfg.DATASETS.SOURCE, merge=cfg.DATASETS.MERGE)

    num_classes_train = [data_loader(cfg,[source], merge=False)[3] for source in cfg.DATASETS.SOURCE]

    # based on input datasets
    bias = (max(num_classes_train))/np.array(num_classes_train)
    bias = bias / bias.sum() * 5

    discriminator_loss=LabelSmoothingLoss(len(sources), weights=bias, smoothing=0.1)
    minus_generator_loss=LabelSmoothingLoss(len(sources), weights=bias, smoothing=0.)
    classification_loss=LabelSmoothingLoss(num_classes, smoothing=0.1)
    from loss.triplet_loss import TripletLoss
    triplet = TripletLoss(cfg.SOLVER.MARGIN)
    triplet_loss=lambda feat, labels: triplet(feat, labels)[0]
    reconstruction_loss = nn.L1Loss()

    module = getattr(generalizers, cfg.MODEL.NAME)
    D = getattr(module, 'Generalizer_D')(len(sources))
    G = getattr(module, 'Generalizer_G')(num_classes, last_stride, pooling)
    if resume:
        checkpoints = get_last_stats(output_dir)
        D.load_state_dict(torch.load(checkpoints[str(type(D))]))
        G.load_state_dict(torch.load(checkpoints[str(type(G))]))
        if device:  # must be done before the optimizer generation
            D.to(device)
            G.to(device)

    discriminator_optimizer = Adam(D.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    generator_optimizer = Adam(G.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    discriminator_scheduler = make_scheduler(cfg, discriminator_optimizer)
    generator_scheduler = make_scheduler(cfg, generator_optimizer)

    base_epo = 0
    if resume:
        discriminator_optimizer.load_state_dict(torch.load(checkpoints['D_opt']))
        generator_optimizer.load_state_dict(torch.load(checkpoints['G_opt']))
        discriminator_scheduler.load_state_dict(torch.load(checkpoints['D_sch']))
        generator_scheduler.load_state_dict(torch.load(checkpoints['G_sch']))
        base_epo = checkpoints['epo']

    # Modify the labels:
    # RULE:
    # according to the order of names in cfg.DATASETS.NAMES, add base numebr

    since = time.time()
    if not resume:
        logger.info("Start training")


    batch_count = 0
    STEP = 4
    Best_R1s = [0, 0, 0, 0]
    Benchmark = [69.6, 43.7, 59.4, 78.2]

    for epoch in range(epochs):
        # anneal = sigmoid(annealing_base + annealing_factor*(epoch+base_epo))
        anneal = max(1 - (1/80 * epoch), 0)
        count = 0
        running_g_loss = 0.
        running_source_loss = 0.
        running_class_acc = 0.
        running_acc_source = 0.
        running_class_loss = 0.
        running_recon_loss = 0.

        reset()

        for data in tqdm(train_loader, desc='Iteration', leave=False):
            # NOTE: zip ensured the shortest dataset dominates the iteration
            D.train()
            G.train()
            images, labels, domains = data
            if device:
                D.to(device)
                G.to(device)
                images, labels, domains = images.to(device), labels.to(device), domains.to(device)

            """Start Training D"""

            feature_vec, scores, gan_vec, recon_vec = G(images)

            for param in G.parameters():
                param.requires_grad = False
            for param in D.parameters():
                param.requires_grad = True

            for _ in range(STEP):
                discriminator_optimizer.zero_grad()

                pred_domain = D([v.detach() for v in gan_vec] if isinstance(gan_vec, list) else gan_vec.detach())  # NOTE: Feat output! Not Probability!

                d_losses, accs = discriminator_loss(pred_domain, domains, compute_acc=True)
                d_source_loss = d_losses.mean()
                d_source_acc = accs.float().mean().item()
                d_loss = d_source_loss

                w_d_loss = anneal * d_loss * gan_d_param

                w_d_loss.backward()
                discriminator_optimizer.step()

            """Start Training G"""

            for param in D.parameters():
                param.requires_grad = False
            for param in G.parameters():
                param.requires_grad = True

            generator_optimizer.zero_grad()

            g_loss = -1. * minus_generator_loss(D(gan_vec), domains).mean()
            class_loss = classification_loss(scores, labels).mean()
            tri_loss = triplet_loss(gan_vec, labels)
            class_loss = class_loss * cfg.SOLVER.LAMBDA1 + tri_loss * cfg.SOLVER.LAMBDA2
            recon_loss = reconstruction_loss(feature_vec, recon_vec)

            w_regularized_g_loss = anneal * gan_g_param * g_loss + class_param * class_loss + recon_loss

            w_regularized_g_loss.backward()
            generator_optimizer.step()

            """Stop training"""

            running_g_loss += g_loss.item()
            running_source_loss += d_source_loss.item()
            running_recon_loss += recon_loss.item()
            running_acc_source += d_source_acc  # TODO: assume all batches are the same size
            running_class_loss += class_loss.item()

            class_acc = (scores.max(1)[1] == labels).float().mean().item()
            running_class_acc += class_acc

            # writer.add_scalar('D_loss', d_source_loss.item(), batch_count)
            # writer.add_scalar('D_acc', d_source_acc, batch_count)
            # writer.add_scalar('G_loss', g_loss.item(), batch_count)
            # writer.add_scalar('Class_loss', class_loss.item(), batch_count)
            # writer.add_scalar('Class_acc', class_acc, batch_count)


            torch.cuda.empty_cache()
            count = count + 1
            batch_count += 1

            # if count == 10:break


        logger.info("Epoch[{}] Iteration[{}] Loss: [G] {:.3f} [D] {:.3f} [Class] {:.3f} [R] {:.3f}, Acc: [Class] {:.3f} [D] {:.3f}, Base Lr: {:.2e}"
                    .format(epoch+base_epo + 1,
                            count,
                            running_g_loss / count,
                            running_source_loss / count,
                            running_class_loss / count,
                            running_recon_loss / count,
                            running_class_acc / count,
                            running_acc_source/ count,
                            generator_scheduler.get_lr()[0]))

        generator_scheduler.step()
        discriminator_scheduler.step()


        if (epoch+base_epo + 1) % checkpoint_period == 0:
            G.cpu()
            G.save(output_dir, epoch+base_epo + 1)
            D.cpu()
            D.save(output_dir, epoch+base_epo + 1)
            torch.save(generator_optimizer.state_dict(), os.path.join(output_dir, 'G_opt_epo'+str(epoch+base_epo+1)+'.pth'))
            torch.save(discriminator_optimizer.state_dict(), os.path.join(output_dir, 'D_opt_epo'+str(epoch+base_epo+1)+'.pth'))
            torch.save(generator_scheduler.state_dict(), os.path.join(output_dir, 'G_sch_epo'+str(epoch+base_epo+1)+'.pth'))
            torch.save(discriminator_scheduler.state_dict(), os.path.join(output_dir, 'D_sch_epo'+str(epoch+base_epo+1)+'.pth'))

        # Validation
        if (epoch+base_epo + 1) % eval_period == 0:
            # Validation on Target Dataset
            for target in cfg.DATASETS.TARGET:
                mAPs = []
                cmcs = []
                for i in range(iteration):

                    set_seeds(i)

                    _, val_loader, num_query, _ = data_loader(cfg,(target,), merge=False, verbose=False)

                    all_feats = []
                    all_pids = []
                    all_camids = []

                    since = time.time()
                    for data in tqdm(val_loader, desc='Feature Extraction', leave=False):
                        G.eval()
                        with torch.no_grad():
                            images, pids, camids = data
                            if device:
                                G.to(device)
                                images = images.to(device)

                            feats = G(images)
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


        # Record Best
        if (epoch+base_epo+1) > 60 and ((epoch+base_epo+1)%5 == 1 or (epoch+base_epo+1)%5 == 2):
            # Validation on Target Dataset
            R1s = []
            for target in cfg.DATASETS.TARGET:
                mAPs = []
                cmcs = []
                for i in range(iteration):

                    set_seeds(i)

                    _, val_loader, num_query, _ = data_loader(cfg,(target,), merge=False, verbose=False)

                    all_feats = []
                    all_pids = []
                    all_camids = []

                    since = time.time()
                    for data in tqdm(val_loader, desc='Feature Extraction', leave=False):
                        G.eval()
                        with torch.no_grad():
                            images, pids, camids = data
                            if device:
                                G.to(device)
                                images = images.to(device)

                            feats = G(images)
                            feats /= feats.norm(dim=-1, keepdim=True)

                        all_feats.append(feats)
                        all_pids.extend(np.asarray(pids))
                        all_camids.extend(np.asarray(camids))

                    cmc, mAP = evaluation(all_feats,all_pids,all_camids,num_query)
                    mAPs.append(mAP)
                    cmcs.append(cmc)

                mAP = np.mean(np.array(mAPs))
                cmc = np.mean(np.array(cmcs), axis=0)
                R1 = cmc[0]
                R1s.append(R1)

            if (np.array(R1s) > np.array(Best_R1s)).all():
                logger.info("Best checkpoint at {}: {}".format(str(epoch+base_epo+1), ', '.join([str(s) for s in R1s])))
                Best_R1s = R1s
                G.cpu()
                G.save(output_dir, -1)
                D.cpu()
                D.save(output_dir, -1)
                torch.save(generator_optimizer.state_dict(), os.path.join(output_dir, 'G_opt_epo'+str(-1)+'.pth'))
                torch.save(discriminator_optimizer.state_dict(), os.path.join(output_dir, 'D_opt_epo'+str(-1)+'.pth'))
                torch.save(generator_scheduler.state_dict(), os.path.join(output_dir, 'G_sch_epo'+str(-1)+'.pth'))
                torch.save(discriminator_scheduler.state_dict(), os.path.join(output_dir, 'D_sch_epo'+str(-1)+'.pth'))
            else:
                logger.info("Rank 1 results: {}".format(', '.join([str(s) for s in R1s])))


    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('-' * 10)

    # writer.close()


if __name__ == "__main__":
    fire.Fire(train)
