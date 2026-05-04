from .lstm import LSTMRegressor
from .transformer import TransformerRegressor
from .cnn import CNNRegressor

ARCHS = {
    'lstm': LSTMRegressor,
    'transformer': TransformerRegressor,
    'cnn': CNNRegressor,
}
