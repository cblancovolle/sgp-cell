from torch import Tensor
from .gpytorch_agent import GPytorchAgent
from .smt_agent import SmtAgent
from .multioutput_smt_agent import MultiOutputSmtAgent

__all__ = ["GPytorchAgent", "SmtAgent", "MultiOutputSmtAgent"]


class Agent:
    def __init__(self, ini_X: Tensor, ini_y: Tensor, **model_kwargs):
        raise NotImplementedError()

    @property
    def current_memory_size(self):
        raise NotImplementedError()

    def predict(self, X_test: Tensor):
        raise NotImplementedError()

    def spatialization(self, eps=1e-6):
        raise NotImplementedError()

    def learn_one(self, x_new: Tensor, y_new: Tensor) -> bool:
        raise NotImplementedError()
