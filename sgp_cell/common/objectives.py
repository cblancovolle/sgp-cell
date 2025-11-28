def compute_aic(n_params, log_likelihood, alpha=0.1, beta=2):
    return alpha * n_params - beta * log_likelihood
