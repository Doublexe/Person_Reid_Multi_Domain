import torchvision.transforms as T
from .RandomErasing import RandomErasing


def transforms(cfg, is_train=True):
    if is_train:
        transform_list = [T.Resize(cfg.INPUT.SIZE_TRAIN),
                        T.RandomHorizontalFlip(p=cfg.INPUT.HF_PROB),
                        T.Pad(cfg.INPUT.PADDING),
                        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                        T.ColorJitter(brightness=cfg.INPUT.BRIGHTNESS, contrast=cfg.INPUT.CONTRAST, saturation=cfg.INPUT.SATURATION, hue=cfg.INPUT.HUE),
                        T.ToTensor(),
                        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)]
        if cfg.INPUT.RE:
            transform_list.append(RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN))
            print(transform_list)
        transform = T.Compose(transform_list)
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])
    
    return transform