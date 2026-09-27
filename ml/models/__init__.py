from .lstm import LSTMRegressor
from .transformer import TransformerRegressor
from .cnn import CNNRegressor

ARCHS = {
    'lstm': LSTMRegressor,
    'transformer': TransformerRegressor,
    'cnn': CNNRegressor,
}


def load_checkpoint(path, device, **model_kwargs):
    """Load a training checkpoint and rebuild its model.

    Returns (model, ckpt). ``model_kwargs`` are forwarded to the architecture
    constructor (input_dim, n_fault_classes, ...); ``rul_scale`` is taken from
    the checkpoint meta. Checkpoints trained before the ``rul_scale`` buffer
    existed predicted raw RUL, so a missing buffer is back-filled with 1.0.
    """
    import torch
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt.get('meta', {})
    rul_scale = float(meta.get('rul_scale', 1.0))
    model = ARCHS[ckpt['arch']](rul_scale=rul_scale, **model_kwargs).to(device)
    state = dict(ckpt['state_dict'])
    state.setdefault('rul_scale', torch.tensor(rul_scale))
    model.load_state_dict(state)
    model.eval()
    return model, ckpt
