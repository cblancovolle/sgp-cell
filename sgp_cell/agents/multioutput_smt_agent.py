import torch
import numpy as np
from torch import Tensor
from smt.surrogate_models import GPX
from typing import List
from sgp_cell.common.objectives import compute_aic


class MultiOutputSmtAgent:
    model: GPX

    def __init__(
        self,
        ini_X: Tensor,
        ini_y: Tensor,
        model_kwargs: dict,
        param_coeff=0.1,
    ):
        ini_X, ini_y = torch.atleast_2d(ini_X), torch.atleast_2d(ini_y)

        self.in_dim, self.out_dim = ini_X.size(-1), ini_y.size(-1)
        self.X_train = ini_X.clone()
        self.y_train = ini_y.clone()
        self.model_kwargs = model_kwargs
        self.param_coef = param_coeff

        self.model = self.build_model(self.X_train, self.y_train)

    @property
    def current_mem_size(self):
        return self.X_train.size(0)

    def build_model(self, X_train: Tensor, y_train: Tensor) -> List[GPX]:
        X_train = X_train.numpy()
        y_train = y_train.numpy()
        models = []
        for i in range(self.out_dim):
            sm = GPX(print_global=False, **self.model_kwargs)
            sm.set_training_values(X_train, y_train[:, i])
            sm.train()
            models += [sm]
        return models

    def _predict(self, models: List[GPX], X_test: np.ndarray):
        pred_values, pred_variances = [], []
        for m in models:
            pred_values += [m.predict_values(X_test)]
            pred_variances += [m.predict_variances(X_test)]
        return np.stack(pred_values, axis=1), np.stack(pred_variances, axis=1)

    def predict(self, X_test: Tensor):
        X_test = X_test.numpy().astype(np.float64)
        posterior_mean, posterior_variance = self._predict(self.model, X_test)
        posterior_mean = torch.as_tensor(posterior_mean)
        posterior_variance = torch.as_tensor(posterior_variance)
        return posterior_mean, posterior_variance

    def spatialization(self, eps=1e-6):
        return (
            torch.mean(self.X_train, dim=0),
            torch.cov(self.X_train.T).view(self.in_dim, self.in_dim)
            + torch.eye(self.in_dim, self.in_dim) * eps,
        )

    def swap_points(self, x_new: Tensor, y_new: Tensor, swap_id):
        swap_mask = torch.ones(self.current_mem_size + 1, dtype=torch.bool)
        swap_mask[swap_id] = False
        self.X_train = torch.vstack((self.X_train, x_new))[swap_mask]
        self.y_train = torch.vstack((self.y_train, y_new))[swap_mask]

    def add_point(self, x_new: Tensor, y_new: Tensor):
        self.X_train = torch.vstack((self.X_train, x_new.view(1, self.in_dim)))
        self.y_train = torch.vstack((self.y_train, y_new.view(1, self.out_dim)))

    def _likelihoods(self, models: List[GPX]):
        return np.stack([m._gpx.likelihoods() for m in models])

    def loss(self, model: List[GPX], X: Tensor, y: Tensor):
        log_likelihood = np.log(self._likelihoods(model)).mean()
        l = compute_aic(X.size(0), log_likelihood, alpha=self.param_coef)
        return l

    def learn_one(self, x_new: Tensor, y_new: Tensor) -> bool:
        d = self.current_mem_size
        X_z, y_z = self.X_train.clone(), self.y_train.clone()  # current configuration
        X_full = torch.vstack((X_z, x_new))
        y_full = torch.vstack((y_z, y_new))

        nominal_loss = self.loss(self.model, X_z, y_z)

        add_model = self.build_model(X_full, y_full)
        add_loss = self.loss(add_model, X_full, y_full)

        _, swap_variances = self.predict(X_z)
        swap_candidate_id = np.argmax(swap_variances.mean(axis=1))
        swap_mask = torch.ones(d + 1, dtype=torch.bool)
        swap_mask[swap_candidate_id] = False
        swap_model = self.build_model(X_full[swap_mask], y_full[swap_mask])
        swap_loss = self.loss(swap_model, X_full[swap_mask], y_full[swap_mask])

        we_could_swap = swap_loss < nominal_loss
        we_could_add = add_loss < nominal_loss
        if we_could_swap and (not we_could_add):
            self.swap_points(x_new, y_new, swap_candidate_id)
            self.model = swap_model
            return True
        if we_could_add and (not we_could_swap):
            self.add_point(x_new, y_new)
            self.model = add_model
            return True
        if we_could_add and we_could_swap:
            we_should_swap = swap_loss < add_loss
            if we_should_swap:
                self.swap_points(x_new, y_new, swap_candidate_id)
                self.model = swap_model
            else:
                self.add_point(x_new, y_new)
                self.model = add_model
            return True
        return False
