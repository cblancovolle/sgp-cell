import torch


def pca(X):
    # X needs to be scaled well and be centered
    U, S, Vt = torch.linalg.svd(X)
    explained_variance = (S**2) / (X.size(0) - 1)
    explained_variance_ratio = explained_variance / torch.sum(explained_variance)
    return (
        Vt,  # (n_features, n_features)
        explained_variance,
        explained_variance_ratio,
    )


def project(X, Vt):
    return X @ Vt.T


def reconstruct(Z, Vt):
    return Z @ Vt
