import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import os
import torch
import models
import fire
from data_loader import data_loader
from utils import check_jupyter_run, get_last_stats
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from utils.seed_utils import set_seeds
if check_jupyter_run():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


RS = 1


def imannotate(x, y, image, ax, zoom=.7, c='blue'):
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    ab = AnnotationBbox(im, (x, y), xycoords='data',
                        bboxprops=dict(edgecolor=c[0], lw=10))
    artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def imscatter(x, y, images):
    # We choose a color palette with seaborn.
    colors = np.unique(y)
    palette = sns.color_palette("hls", colors.size)
    palette = {colors[i]: p for i, p in enumerate(palette)}

    def get_color(y_i): return palette[y_i]
    get_color = np.vectorize(get_color)

    # We create a scatter plot.
    f = plt.figure(figsize=(100, 100))
    ax = plt.subplot(aspect='equal')
    for i in range(x.shape[0]):
        imannotate(x[i, 0], x[i, 1], images[i], ax, c=np.array(get_color(
            y[i])).reshape([1, -1]))
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    return f, ax


def scatter(x, y):
    # We choose a color palette with seaborn.
    colors = np.unique(y)
    palette = sns.color_palette("hls", colors.size)
    palette = {colors[i]: p for i, p in enumerate(palette)}

    def get_color(y_i): return palette[y_i]
    get_color = np.vectorize(get_color)

    # We create a scatter plot.
    f = plt.figure(figsize=(20, 20))
    ax = plt.subplot(aspect='equal')
    for i in range(x.shape[0]):
        sc = ax.scatter(x[i, 0], x[i, 1], lw=0, s=150, c=np.array(get_color(
            y[i])).reshape([1, -1]), alpha=0.8, marker='o' if 'target' in y[i] else '^')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in colors:
        # Position of each label.
        xtext, ytext = np.median(x[y == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def unique_by_pid(pids, *tensors):
    unique = torch.unique(pids)
    results = [unique]
    for tensor in tensors:
        temp = []
        for p in unique:
            temp.append(tensor[pids==p][0:1])
        results.append(torch.cat(temp))
    return results



def visualize(config_file,
              model_type="generalizer",
              mode="img",
              num_batch=2,
              metric='euclidean',
              vis_options=['train'],
              perplexity=30,
              save_dir='./',
              **kwargs):

    """ Visualize the feature vector space  with options.

    CAUTION: TSNE is sensitive to its paramters and all potential clusters.
             That is why I use unique_by_pid function. Do not use exact mode for tsne.
             See https://distill.pub/2016/misread-tsne/ for details.

    Parameter
    ---------
    config_file : str=yacs.CfgNode
    model_type : str
        generalizer or normal models
    mode : str
        to plot dots/images/etc on tsne visualizatioin
    num_batch : int
        number of batches from data used to visualize
    metric : str
        distance metric for tsne. cosine or euclidean
    vis_options : List[str]
        to visualize on train/test data
    perplexity : int
        a critical parameter for tsne. Try multiple to see effects.

    Return
    ---------
    save a png picture to save_dir

    """

    if mode not in ["img", "dot"]:
        raise ValueError("Mode can only be img or dot.")

    if model_type == "normal":
        from config.default_multi_domain import _C as cfg
    elif model_type == "generalizer":
        from config.default_multi_domain import _C as cfg
    else:
        raise ValueError("Model type can only be normal or generalizer.")

    cfg.merge_from_file(config_file)
    if kwargs:
        opts = []
        for k, v in kwargs.items():
            opts.append(k)
            opts.append(v)
        cfg.merge_from_list(opts)
    cfg.freeze()

    def recover_image(image):
        # Final format is still Tensor.
        image = image.permute(
            0, 2, 3, 1) * torch.Tensor(cfg.INPUT.PIXEL_STD) + torch.Tensor(cfg.INPUT.PIXEL_MEAN)
        return image

    # PersonReID_Dataset_Downloader('./datasets',cfg.DATASETS.NAMES)
    _, _, _, num_classes = data_loader(
        cfg, cfg.DATASETS.SOURCE, merge=cfg.DATASETS.MERGE, verbose=False)

    device = torch.device(cfg.DEVICE)

    if model_type == "normal":
        model = getattr(models, cfg.MODEL.NAME)(
            num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.POOL)
        checkpoints = get_last_stats(cfg.OUTPUT_DIR, [cfg.MODEL.NAME])
        model_dict = torch.load(checkpoints[cfg.MODEL.NAME])
        model.load_state_dict(model_dict)

    elif model_type == "generalizer":
        import generalizers
        module = getattr(generalizers, cfg.MODEL.NAME)
        G = getattr(module, 'Generalizer_G')(
            num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.POOL)
        checkpoints = get_last_stats(cfg.OUTPUT_DIR, [str(
            type(G)), 'D_opt', 'G_opt', 'D_sch', 'G_sch', 'epo'])
        G.load_state_dict(torch.load(checkpoints[str(type(G))]))
        if device:  # must be done before the optimizer generation
            G.to(device)
        model = G

    model = model.eval()

    x = []
    y = []
    imgs = []

    NUM = num_batch

    if 'test' in vis_options:
        test_val_stats = [data_loader(cfg, (target,), merge=False, verbose=False)[
            1] for target in cfg.DATASETS.TARGET]

        for i, val_loader in enumerate(tqdm.tqdm(test_val_stats, desc="")):
            count = 0
            for data in val_loader:
                model.eval()
                with torch.no_grad():
                    images, pids, camids = data
                    pids, images, camids = unique_by_pid(pids, images, camids)
                    imgs.append(recover_image(images).data.numpy())
                    if device:
                        model.to(device)
                        images = images.to(device)
                    feats = model(images)
                    if metric=='cosine':
                        feats /= feats.norm(dim=-1, keepdim=True)
                    x.append(feats.data.cpu().numpy())
                    y.extend([cfg.DATASETS.TARGET[i] + '_T' for e in pids])
                count += 1
                if count == NUM:
                    break

    if 'train' in vis_options:
        train_val_stats = [data_loader(cfg, (source,), merge=False, verbose=False)[
            0] for source in cfg.DATASETS.SOURCE]

        for i, train_loader in enumerate(train_val_stats):
            count = 0
            for data in train_loader:
                model.eval()
                with torch.no_grad():
                    images, pids, camids = data
                    pids, images, camids = unique_by_pid(pids, images, camids)
                    imgs.append(recover_image(images).data.numpy())
                    if device:
                        model.to(device)
                        images = images.to(device)

                    feats = model(images)
                    if metric=='cosine':
                        feats /= feats.norm(dim=-1, keepdim=True)
                    x.append(feats.data.cpu().numpy())
                    y.extend([cfg.DATASETS.SOURCE[i] + '_S' for e in pids])
                count += 1
                if count == NUM:
                    break

    set_seeds(1)

    X = np.concatenate(x, 0)
    y = np.array(y)
    imgs = np.concatenate(imgs, 0)

    digits_proj = TSNE(random_state=RS, perplexity=perplexity, metric=metric).fit_transform(X)

    if mode=="img":
        imscatter(digits_proj, y, imgs)
    elif mode=="dot":
        scatter(digits_proj, y)
    else:
        raise ValueError("Mode can only be img or dot.")

    plt.savefig(os.path.join(save_dir, 'tsne-' + mode+ '-' + metric + '_' + cfg.MODEL.NAME + '.png'), dpi=120)


if __name__ == '__main__':
    fire.Fire(visualize)
