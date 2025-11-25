import torch
import numpy as np

from sgp_cell.common.covariance import mahalanobis2
from torch import Tensor
from collections import deque
from scipy.stats import chi2
from sgp_cell.agents import Agent
from sgp_cell.common.pca import pca, project, reconstruct


class PCAOnlineTrainer:
    def __init__(
        self,
        in_dim,
        out_dim,
        agent_cls,
        agent_kwargs,
        neighbor_confidence=0.95,
        min_points=3,
        k_components=3,
        reconstruct_threshold=0.2,
    ):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.agent_cls = agent_cls
        self.agent_kwargs = agent_kwargs
        self.min_points = min_points
        self.neighbor_confidence = neighbor_confidence
        self.mahalanobis_neighbor_confidence = torch.as_tensor(
            chi2.ppf(neighbor_confidence, df=k_components)
        )
        self.buffer = deque([], maxlen=min_points)
        self.agents: list[Agent] = []

        self.k_components = k_components
        self.reconstruct_threshold = reconstruct_threshold
        self.var = torch.empty((0, in_dim))  # for standardization
        self.mean = torch.empty((0, in_dim))  # for standardization
        self.pca_var = torch.empty((0, k_components))  # for projection
        self.pca_mean = torch.empty((0, k_components))  # for projection
        self.pca_Vt = torch.empty((0, k_components, in_dim))  # for projection

    @property
    def n_agents(self):
        return len(self.agents)

    @property
    def buffer_data(self):
        X, y = map(torch.vstack, zip(*self.buffer))
        return X.double(), y.double()

    def reset(self):
        self.previous_neighbors = []
        self.buffer.clear()

    def create_agent(self, ini_X: Tensor, ini_y: Tensor):
        new_agent = self.agent_cls(
            ini_X,
            ini_y,
            **self.agent_kwargs,
        )
        mean, cov = new_agent.spatialization()
        var = torch.diag(cov)
        # for standardization
        self.var = torch.vstack(
            [
                self.var,
                var.view(1, self.in_dim),
            ]
        )
        self.mean = torch.vstack(
            [
                self.mean,
                mean.view(1, self.in_dim),
            ]
        )
        # for projection
        X_train = new_agent.X_train
        X_train_standardized = (X_train - mean) / (torch.sqrt(var) + 1e-8)
        Vt, explained_variance, explained_variance_ratio = pca(X_train_standardized)
        projected_X_train = project(X_train_standardized, Vt[: self.k_components])

        projected_mean = torch.mean(projected_X_train, dim=0)
        self.pca_var = torch.vstack(
            [
                self.pca_var,
                explained_variance[: self.k_components].view(1, self.k_components),
            ]
        )
        self.pca_mean = torch.vstack(
            [
                self.pca_mean,
                projected_mean.view(1, self.k_components),
            ]
        )
        self.pca_Vt = torch.vstack(
            [
                self.pca_Vt,
                Vt[: self.k_components].view(1, self.k_components, self.in_dim),
            ]
        )

        self.agents.append(new_agent)
        self.buffer.clear()

    def project(self, x_new, agents_idxs=slice(None)):
        # standardize
        x_new_standardized = (x_new - self.mean[agents_idxs]) / (
            torch.sqrt(self.var[agents_idxs]) + 1e-8
        )
        # project
        x_new_projected = torch.vmap(project)(
            x_new_standardized, self.pca_Vt[agents_idxs]
        )
        return x_new_projected  # (n_agents, k_components)

    def distances(self, x_new, agents_idxs=slice(None)):
        x_new_projected = self.project(x_new)
        # ==== MAHALANOBIS ====
        feature_P = torch.linalg.inv(torch.vmap(torch.diag)(self.pca_var[agents_idxs]))
        dm2 = torch.vmap(mahalanobis2)(
            torch.as_tensor(x_new_projected).view(-1, self.k_components),
            feature_P,
            self.pca_mean[agents_idxs],
        )  # ()
        return dm2

    def learn_one(self, x_new: Tensor, y_new: Tensor):
        self.buffer.append((x_new, y_new))
        if len(self.buffer) < self.min_points:
            return {}
        if self.n_agents == 0:
            self.create_agent(*self.buffer_data)
            self.previous_neighbors = torch.as_tensor(
                [self.n_agents - 1], dtype=torch.long
            )
            return {}

        # check neighbors
        x_new_projected = self.project(x_new)  # (n_agents, k_components)
        reconstructed_X_new = torch.vmap(reconstruct)(
            x_new_projected, self.pca_Vt[:, : self.k_components]
        )  # (n_agents, in_dim)
        relative_reconstruction_error = (
            (reconstructed_X_new - x_new) / (reconstructed_X_new + 1e-6)
        ).mean(
            dim=1
        )  # (n_agents,)
        reconstruct_mask = relative_reconstruction_error < self.reconstruct_threshold
        distances = self.distances(x_new.view(1, -1))
        neighbors_mask = (distances <= self.mahalanobis_neighbor_confidence).view(
            self.n_agents
        ) & reconstruct_mask
        neighbors = torch.where(neighbors_mask)[0]

        if len(neighbors) == 0:
            agents_to_update = self.previous_neighbors
        else:
            agents_to_update = neighbors

        if len(agents_to_update) > 0:
            agents_to_update = [
                agents_to_update[torch.argmin(distances[agents_to_update])]
            ]
        # update agents
        point_has_been_ingested = False
        for id in agents_to_update:
            swapped = self.agents[id].learn_one(x_new, y_new)
            point_has_been_ingested |= swapped

            if swapped:
                self.mean[id], cov = self.agents[id].spatialization()
                self.var[id] = torch.diag(cov)
                X_train = self.agents[id].X_train
                X_train_standardized = (X_train - self.mean[id]) / (
                    torch.sqrt(self.var[id]) + 1e-8
                )
                Vt, explained_variance, explained_variance_ratio = pca(
                    X_train_standardized
                )
                projected_X_train = project(
                    X_train_standardized, Vt[: self.k_components]
                )
                projected_mean = torch.mean(projected_X_train, dim=0)
                self.pca_var[id] = explained_variance[: self.k_components]
                self.pca_mean[id] = projected_mean
                self.pca_Vt[id] = Vt[: self.k_components]

        self.previous_neighbors = agents_to_update

        if (not point_has_been_ingested) and (len(neighbors) == 0):
            self.create_agent(*self.buffer_data)
            self.previous_neighbors = torch.as_tensor(
                [self.n_agents - 1], dtype=torch.long
            )

        return dict(
            n_agents=self.n_agents,
            n_neighbors=len(neighbors),
            n_updated=len(agents_to_update),
            point_ingested=point_has_been_ingested,
            min_mahalanobis_dist=distances.min().numpy(),
            neighbors=neighbors.numpy(),
            agents_selected=np.array(agents_to_update),
        )

    def predict_one(self, x: Tensor, k=4):
        # check k closest
        distances = self.distances(x.view(1, -1)).view(-1)
        _, closest_agents = torch.topk(
            distances, k=min(k, self.n_agents), largest=False
        )

        # predict
        mus, vars = [], []
        for agent_id in closest_agents.view(-1):
            mu, var = self.agents[agent_id.item()].predict(x)
            mus += [mu]
            vars += [var]
        mus = torch.vstack(mus)
        vars = torch.vstack(vars)

        # check for most confident agent
        most_confident_idx = torch.argmin(vars.mean(dim=1))
        return mus[most_confident_idx], vars[most_confident_idx]
