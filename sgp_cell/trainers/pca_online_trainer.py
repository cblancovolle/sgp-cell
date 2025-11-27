import torch
import numpy as np

from sgp_cell.common.covariance import mahalanobis2
from torch import LongTensor, Tensor
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
        confidence_forget_lambd=0.2,
        confidence_norm_steepness=2,
        confidence_destroy_th=0.1,
    ):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.agent_cls = agent_cls
        self.agent_kwargs = agent_kwargs
        self.min_points = min_points
        self.confidence_forget_lmbd = confidence_forget_lambd
        self.confidence_norm_steepness = confidence_norm_steepness
        self.confidence_destroy_threshold = confidence_destroy_th
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
        self.confidence = torch.empty((0, 1))

    @property
    def n_agents(self):
        return len(self.agents)

    @property
    def buffer_data(self):
        X, y = map(torch.vstack, zip(*self.buffer))
        return X.double(), y.double()

    def reset(self):
        self.buffer.clear()

    def destroy_agent(self, agent_idxs: LongTensor):
        mask = ~torch.isin(torch.arange(self.n_agents), agent_idxs)
        self.var = self.var[mask]
        self.mean = self.mean[mask]
        self.pca_var = self.pca_var[mask]
        self.pca_mean = self.pca_mean[mask]
        self.pca_Vt = self.pca_Vt[mask]
        self.confidence = self.confidence[mask]
        self.agents = [a for id, a in enumerate(self.agents) if id not in agent_idxs]

    def create_agent(self, ini_X: Tensor, ini_y: Tensor):
        new_agent = self.agent_cls(
            ini_X,
            ini_y,
            **self.agent_kwargs,
        )
        self.confidence = torch.vstack(
            [
                self.confidence,
                torch.zeros((1, 1)),
            ]
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

    def reconstruction_error(self, x_new):
        x_new_projected = self.project(x_new)  # (n_agents, k_components)
        reconstructed_X_new_standardized = torch.vmap(reconstruct)(
            x_new_projected, self.pca_Vt[:, : self.k_components]
        )  # (n_agents, in_dim)
        reconstructed_X_new = (
            reconstructed_X_new_standardized * torch.sqrt(self.var) + 1e-8 + self.mean
        )
        relative_reconstruction_error = torch.norm(
            reconstructed_X_new - x_new, dim=1
        ) / (torch.norm(x_new) + 1e-8)
        return relative_reconstruction_error

    def learn_one(self, x_new: Tensor, y_new: Tensor):
        if self.n_agents == 0 and (len(self.buffer) >= self.min_points):
            self.create_agent(*self.buffer_data)
            return dict(
                n_agents=self.n_agents,
                n_neighbors=0,
                n_updated=1,
                n_created=1,
                point_ingested=False,
                min_mahalanobis_dist=np.nan,
                neighbors=[],
                agents_selected=[],
            )
        elif self.n_agents == 0:
            self.buffer.append((x_new, y_new))
            return dict(
                n_agents=self.n_agents,
                n_neighbors=0,
                n_updated=0,
                n_created=0,
                point_ingested=False,
                min_mahalanobis_dist=np.nan,
                neighbors=[],
                agents_selected=[],
            )

        # check neighbors
        relative_reconstruction_error = self.reconstruction_error(x_new)
        reconstruct_mask = relative_reconstruction_error < self.reconstruct_threshold
        distances = self.distances(x_new.view(1, -1))
        neighbors_mask = (distances <= self.mahalanobis_neighbor_confidence).view(
            self.n_agents
        ) & reconstruct_mask
        neighbors = torch.where(neighbors_mask)[0]
        n_neighbors = len(neighbors)
        closest = torch.argmin(distances)

        if n_neighbors == 0:
            agents_to_update = [closest]  # TODO: best reconstructed
        else:
            agents_to_update = neighbors.clone()

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

        # update confidence
        if n_neighbors > 1:
            n_mus, _ = self._predict_one(x_new, neighbors)
            activations = torch.exp(-0.5 * distances[neighbors]).view(-1, 1)

            # weighted mean
            total_activation = torch.sum(activations) + 1e-12
            weights = activations / total_activation
            w_mu = torch.sum(n_mus * weights, dim=0)
            E = torch.abs(y_new - w_mu).mean()  # (out_dim,)

            Emi = []
            for i in range(len(neighbors)):
                mask = torch.arange(n_neighbors) != i
                total_activation_mi = torch.sum(activations[mask]) + 1e-12
                weights_mi = activations[mask] / total_activation_mi
                w_mu_mi = torch.sum(n_mus[mask] * weights_mi, dim=0)
                Emi += [torch.abs(y_new.view(-1) - w_mu_mi).mean()]

            Emi = torch.stack(Emi)  # (n_neighbors,)
            Ci = torch.tanh(self.confidence_norm_steepness * ((Emi - E) / E))
            # update confidence
            lmb = self.confidence_forget_lmbd
            self.confidence[neighbors] = (
                self.confidence[neighbors] * (1 - lmb) + lmb * Ci.view(-1, 1)
            ).float()

            # destroy too bad agents
            agents_to_destroy = neighbors[
                (
                    self.confidence[neighbors] < -self.confidence_destroy_threshold
                ).squeeze()
            ]
            self.destroy_agent(agents_to_destroy)

        if (
            (not point_has_been_ingested)
            and (n_neighbors == 0)
            and (len(self.buffer) >= self.min_points)
        ):
            self.create_agent(*self.buffer_data)
        if n_neighbors == 0 and (not point_has_been_ingested):
            self.buffer.append((x_new, y_new))

        return dict(
            n_agents=self.n_agents,
            n_neighbors=n_neighbors,
            n_updated=len(agents_to_update),
            n_created=int((not point_has_been_ingested) and (len(neighbors) == 0)),
            point_ingested=point_has_been_ingested,
            min_mahalanobis_dist=distances.min().numpy(),
            neighbors=neighbors.numpy(),
            agents_selected=np.array(agents_to_update),
        )

    def _predict_one(self, x: Tensor, agents_idxs: LongTensor):
        mus, vars = [], []
        for agent_id in agents_idxs.view(-1):
            mu, var = self.agents[agent_id.item()].predict(x)
            mus += [mu]
            vars += [var]
        mus = torch.vstack(mus)
        vars = torch.vstack(vars)
        return mus, vars

    def predict_one(self, x: Tensor, k=4):
        # check k closest
        distances = self.distances(x.view(1, -1)).view(-1)
        closest_values, closest_agents = torch.topk(
            distances, k=min(k, self.n_agents), largest=False
        )
        activations = torch.exp(-0.5 * closest_values).view(-1, 1)

        # predict
        mus, vars = self._predict_one(x, closest_agents)

        # print(mus.shape, vars.shape, activations.shape)
        # weighted mean
        # total_activation = torch.sum(activations)
        # w_mu = torch.sum(mus * activations, dim=0) / total_activation
        # w_var = (
        #     torch.sum(activations * (vars + (mus - w_mu) ** 2), dim=0)
        #     / total_activation
        # )
        # return w_mu, w_var

        # check for most confident agent
        most_confident_idx = torch.argmin(vars.mean(dim=1))
        return mus[most_confident_idx], vars[most_confident_idx]
