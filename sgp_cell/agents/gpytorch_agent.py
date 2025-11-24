from typing import Literal
import torch
from torch import Tensor
from gpytorch.kernels import RBFKernel
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from sgp_cell.common.objectives import compute_aic
from botorch.optim.stopping import ExpMAStoppingCriterion


class GPytorchAgent:
    model: SingleTaskGP

    def __init__(
        self,
        ini_X: Tensor,
        ini_y: Tensor,
        max_train_iter=50,
        opt: Literal["scipy", "torch"] = "scipy",
        likelihood_fn: Literal["looll", "ll"] = "ll",
        device="cpu",
    ):
        self.device = device
        self.opt = opt

        ini_X, ini_y = torch.atleast_2d(ini_X), torch.atleast_2d(ini_y)
        self.in_dim, self.out_dim = ini_X.size(-1), ini_y.size(-1)
        self.X_train = ini_X.clone().to(self.device)
        self.y_train = ini_y.clone().to(self.device)
        self.max_train_iter = max_train_iter

        if likelihood_fn == "ll":
            self.likelihood_fn = ExactMarginalLogLikelihood
        elif likelihood_fn == "looll":
            self.likelihood_fn = LeaveOneOutPseudoLikelihood
        else:
            raise ValueError(
                "Unknown likelihood function argument provided. Try seting values in [looll, ll]"
            )

        self.model = self.build_model(self.X_train, self.y_train)

    @property
    def current_mem_size(self):
        return self.X_train.size(0)

    def build_model(self, X_train: Tensor, y_train: Tensor) -> SingleTaskGP:
        base_kernel = RBFKernel(ard_num_dims=self.in_dim)
        model = SingleTaskGP(
            X_train,
            y_train,
            covar_module=base_kernel,
        )
        model.train()
        mll = self.likelihood_fn(model.likelihood, model)
        if self.opt == "scipy":
            fit_gpytorch_mll_scipy(mll, options={"maxiter": self.max_train_iter})
        elif self.opt == "torch":
            fit_gpytorch_mll_torch(
                mll,
                stopping_criterion=ExpMAStoppingCriterion(maxiter=self.max_train_iter),
            )
        else:
            raise ValueError(
                "Unknown opt argument provided. Try setting values in [scipy, torch]"
            )
        model.eval()
        return model

    def predict(self, X_test):
        X_test = X_test.to(self.device)
        posterior = self.model.posterior(X_test.view(-1, self.in_dim))
        return posterior.mean, posterior.variance

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

    def loss(self, model: SingleTaskGP, X: Tensor, y: Tensor):
        with torch.no_grad():
            model.train()
            mll = self.likelihood_fn(model.likelihood, model)
            output = model(*model.train_inputs)  # always use exact training inputs
            log_likelihood = mll(output, model.train_targets).mean(dim=0)
            model.eval()
            l = compute_aic(X.size(0), log_likelihood, alpha=2, beta=2)
        return l

    def learn_one(self, x_new: Tensor, y_new: Tensor) -> bool:
        x_new, y_new = x_new.to(self.device), y_new.to(self.device)
        d = self.current_mem_size
        X_z, y_z = self.X_train.clone(), self.y_train.clone()  # current configuration
        X_full = torch.vstack((X_z, x_new))
        y_full = torch.vstack((y_z, y_new))

        nominal_loss = self.loss(self.model, X_z, y_z)

        add_model = self.build_model(X_full, y_full)
        add_loss = self.loss(add_model, X_full, y_full)

        swap_variances = self.model.posterior(X_z).variance.mean(dim=1)
        swap_candidate_id = torch.argmax(swap_variances.view(-1))
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
