import sys
from file_parse import func_parse
sys.path.append('..')

from PIL import Image
from data_loader.transforms.ColorBalance import ColorBalance


if __name__ == "__main__":
    cb = ColorBalance(satLevel=0.1)
    func = lambda pth: cb(Image.open(pth))
    save_func = lambda pth, img: img.save(pth)

    func_parse('./reid_samples', './color_balance_test', func, save_func)
