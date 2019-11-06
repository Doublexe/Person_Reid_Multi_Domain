import sys
from file_parse import func_parse
sys.path.append('..')

from PIL import Image
from data_loader.transforms.HistEqualization import HistEqualization


if __name__ == "__main__":
    he = HistEqualization()
    func = lambda pth: he(Image.open(pth))
    save_func = lambda pth, img: img.save(pth)

    func_parse('./reid_samples', './hist_equal_test', func, save_func)
