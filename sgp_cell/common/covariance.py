import torch
from torch.distributions import MultivariateNormal, Chi2


def downdate_covariance(C, mu, n, x):
    mu_new = (n * mu - x) / (n - 1)
    delta = x - mu
    C_new = ((n - 2) / (n - 1)) * C - (n / ((n - 1) ** 2)) * torch.outer(delta, delta)
    return C_new, mu_new


batch_downdate_covariance = torch.vmap(downdate_covariance, in_dims=(0, 0, 0, None))


def update_covariance_welford(cov, mu, n, x):
    n_new = n + 1
    delta = x - mu
    mu_new = mu + delta / n_new
    delta2 = x - mu_new

    C = cov * (n - 1).clip(1)
    C_new = C + torch.outer(delta, delta2)
    return C_new / (n_new - 1).clip(1), mu_new


batch_update_covariance_welford = torch.vmap(
    update_covariance_welford, in_dims=(0, 0, 0, None)
)


def weighted_update_covariance_welford(cov, mu, n, weight, x):
    n_new = n + weight
    delta = x - mu
    mu_new = mu + (delta * weight) / n_new
    delta2 = x - mu_new

    C = cov * (n - 1).clip(1)
    C_new = C + weight * torch.outer(delta, delta2)
    return C_new / (n_new - 1).clip(1), mu_new


batch_weighted_update_covariance_welford = torch.vmap(
    weighted_update_covariance_welford, in_dims=(0, 0, 0, 0, None)
)


def mahalanobis2(x, P, mu):
    """Compute squared mahalanobis distance

    Args:
        x (Tensor): (n_dim,)
        P (Tensor): precision matrix (n_dim, n_dim)
        mu (Tensor): mean (state_dim)

    Returns:
        Tensor: (1,)
    """
    return (x - mu).T @ P @ (x - mu)


batch_mahalanobis2 = torch.vmap(mahalanobis2, in_dims=(None, 0, 0))


def batch_mvn_logprob(means, covs, X):
    """Compute logprobs for a bunch of mvn for a bunch of samples

    Args:
        means (Tensor): (b1_size, D)
        covs (Tensor): (b1_size, D, D)
        X (Tensor): (b2_size, D)

    Returns:
        Tensor: (b1_size, b2_size)
    """
    N, D = means.size()
    B = X.size(0)
    means_expanded = means[:, None, :].expand(N, B, D)
    covs_expanded = covs[:, None, :, :].expand(N, B, D, D)
    samples_expanded = X[None, :, :].expand(N, B, D)

    means_flat = means_expanded.reshape(N * B, D)
    covs_flat = covs_expanded.reshape(N * B, D, D)
    samples_flat = samples_expanded.reshape(N * B, D)

    mvn = MultivariateNormal(means_flat, covs_flat)
    log_probs_flat = mvn.log_prob(samples_flat)  # [N*B]
    log_probs = log_probs_flat.reshape(N, B)  # [N, B]
    return log_probs


def project_to_pd_cone(A, eps=1e-6):
    # Eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(A)

    # Clip eigenvalues to ensure they're positive
    eigvals_clipped = torch.clamp(eigvals, min=eps)

    # Reconstruct the matrix
    A_pd = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T

    return A_pd


def batch_gaussian_similarity(means, covs, X):
    """
    Compute similarity scores from Mahalanobis distance using Chi-squared CDF.

    Args:
        means (Tensor): (N, D) - means of the Gaussians
        covs (Tensor): (N, D, D) - covariances of the Gaussians
        X (Tensor): (B, D) - input samples

    Returns:
        Tensor: (N, B) similarity scores in [0, 1]
    """
    N, D = means.size()
    B = X.size(0)

    # Expand tensors to match shapes: (N, B, D)
    means_exp = means[:, None, :].expand(N, B, D)
    covs_exp = covs[:, None, :, :].expand(N, B, D, D)
    X_exp = X[None, :, :].expand(N, B, D)

    diffs = X_exp - means_exp  # (N, B, D)

    # Compute inverse of covariance matrices (N, B, D, D)
    covs_flat = covs_exp.reshape(N * B, D, D)
    inv_covs_flat = torch.linalg.inv(covs_flat)
    inv_covs = inv_covs_flat.view(N, B, D, D)

    # Compute Mahalanobis squared: (N, B)
    diffs_unsq = diffs.unsqueeze(-1)  # (N, B, D, 1)
    left = torch.matmul(inv_covs, diffs_unsq)  # (N, B, D, 1)
    mahal_sq = torch.matmul(diffs.unsqueeze(2), left).squeeze(-1).squeeze(-1)  # (N, B)

    # Chi-squared distribution with D degrees of freedom
    chi2 = Chi2(df=D)
    similarity = 1 - chi2.cdf(mahal_sq)

    return similarity  # (N, B)


def bhattacharyya_coeff(mu1, cov1, mu2, cov2):
    cov_avg = (cov1 + cov2) / 2

    # Calculate determinants
    log_det_cov1 = torch.logdet(cov1)
    log_det_cov2 = torch.logdet(cov2)
    log_det_cov_avg = torch.logdet(cov_avg)

    # Compute inverse of average covariance
    L = torch.linalg.cholesky(cov_avg)
    inv_cov_avg = torch.cholesky_inverse(L)
    # inv_cov_avg = torch.linalg.inv(cov_avg)

    # Quadratic form term
    term = 0.125 * mahalanobis2(mu1, inv_cov_avg, mu2)

    # Bhattacharyya coefficient
    coeff = torch.exp(
        0.5 * log_det_cov_avg - 0.25 * (log_det_cov1 + log_det_cov2) - term
    )

    return coeff.squeeze()


def bhattacharyya_distance(mu1, cov1, mu2, cov2):
    return -torch.log(bhattacharyya_coeff(mu1, cov1, mu2, cov2))


def kl_divergence(mu1, cov1, mu2, cov2):
    n = mu1.size(-1)
    # Calculate log determinant of covariance matrices
    log_det_cov1 = torch.logdet(cov1)
    log_det_cov2 = torch.logdet(cov2)

    # Trace term
    inv_cov2 = torch.inverse(cov2)
    trace_term = (
        torch.trace(torch.matmul(inv_cov2, cov1))
        if cov1.dim() == 2
        else torch.einsum("bij,bji->b", inv_cov2, cov1)
    )

    # Difference in means
    diff = mu2 - mu1
    if cov1.dim() == 2:
        mahalanobis = torch.dot(diff, torch.matmul(inv_cov2, diff))
    else:
        mahalanobis = torch.einsum("bi,bij,bj->b", diff, inv_cov2, diff)

    kl = 0.5 * (log_det_cov2 - log_det_cov1 - n + trace_term + mahalanobis)
    return kl


def jeffrey_divergence(mu1, cov1, mu2, cov2):
    return 0.5 * (
        kl_divergence(mu1, cov1, mu2, cov2) + kl_divergence(mu2, cov2, mu1, cov1)
    )


def toeplitz_symmetric(
    A,  # (b_size, n_dim, n_dim)
):
    # https://en.wikipedia.org/wiki/Toeplitz_matrix
    return 0.5 * (A + A.transpose(-1, -2))
