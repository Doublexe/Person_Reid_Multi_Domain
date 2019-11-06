import fire
import torch
import models
import generalizers
from data_loader import data_loader
from utils import get_last_stats


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def trace(config_file, model_type="generalizer", **kwargs):

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

    # PersonReID_Dataset_Downloader('./datasets',cfg.DATASETS.NAMES)
    train_loader, _, _, num_classes = data_loader(
        cfg, cfg.DATASETS.SOURCE, merge=cfg.DATASETS.MERGE)

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

    input_names = ['input']
    output_names = ['output']

    batch = 1
    images = torch.randn(batch, 3, 256, 128, requires_grad=True)

    if device:
        model.to(device)
        images = images.to(device)

    torch.onnx.export(model, images, 'test.onnx', verbose=True, input_names=input_names,
                      output_names=output_names)  #, dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})

    # import onnxruntime
    #
    # ort_session = onnxruntime.InferenceSession("test.onnx")
    #
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    #
    # # compute ONNX Runtime output prediction
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images[:32])}
    # ort_outs = ort_session.run(None, ort_inputs)
    # print(model(images[:32]))
    # print(ort_session.get_inputs())
    # print(ort_outs)


if __name__ == '__main__':
    fire.Fire(trace)
