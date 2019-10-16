from data_loader.datasets_importer import __factory

names = [name for name in __factory.keys()]

def get_dataset_name(pth):
    ls = pth.split(r'/')
    for name in names:
        if name in ls:
            return name
    return None
