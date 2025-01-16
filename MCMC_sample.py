import numpy as np
import matplotlib.pyplot as plt
import pandas

# Simulated data
np.random.seed(42)
n = 1000  # Number of observations
true_mu = 2.0  # True mean
sigma = 1.0  # Known standard deviation
data = np.random.normal(true_mu, sigma, n)


# Metropolis-Hastings MCMC
def log_likelihood(mu, data, sigma):
    """Log-likelihood of the data given mu."""
    return -0.5 * np.sum((data - mu) ** 2) / sigma ** 2


def log_prior(mu):
    """Log-prior for mu (standard normal)."""
    return -0.5 * mu ** 2


def log_posterior(mu, data, sigma):
    """Log-posterior is proportional to log-likelihood + log-prior."""
    return log_likelihood(mu, data, sigma) + log_prior(mu)


# MCMC parameters
n_samples = 5000
mu_current = 0.0  # Starting value
samples = []
acceptances = 0
proposal_std = 0.5  # Standard deviation of the proposal distribution

# MCMC sampling
for _ in range(n_samples):
    mu_proposed = np.random.normal(mu_current, proposal_std)

    # Compute acceptance probability
    log_accept_ratio = log_posterior(mu_proposed, data, sigma) - log_posterior(mu_current, data, sigma)
    accept_prob = np.exp(log_accept_ratio)

    # Accept/reject step
    if np.random.rand() < accept_prob:
        mu_current = mu_proposed
        acceptances += 1

    samples.append(mu_current)

# Convert samples to NumPy array
samples = np.array(samples)

# Acceptance rate
acceptance_rate = acceptances / n_samples
print(f"Acceptance rate: {acceptance_rate:.2f}")

# Posterior mean and credible interval
posterior_mean = np.mean(samples)
credible_interval = np.percentile(samples, [2.5, 97.5])

print(f"Posterior mean: {posterior_mean:.2f}")
print(f"95% credible interval: {credible_interval}")

# Plotting the results
plt.figure(figsize=(10, 6))

# Trace plot
plt.subplot(2, 1, 1)
plt.plot(samples, lw=0.5)
plt.title("Trace Plot")
plt.ylabel("mu")

# Posterior distribution
plt.subplot(2, 1, 2)
plt.hist(samples, bins=30, density=True, alpha=0.7, label="Posterior samples")
plt.axvline(posterior_mean, color="red", linestyle="--", label="Posterior mean")
plt.title("Posterior Distribution")
plt.xlabel("mu")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()

print(pandas.DataFrame(samples))
pandas.DataFrame(samples).to_clipboard()

import numpy as np
import matplotlib.pyplot as plt

# Simulated data
np.random.seed(42)
n = 1000  # Number of observations
true_mu = 2.0  # True mean
sigma = 1.0  # Known standard deviation
data = np.random.normal(true_mu, sigma, n)

# Gibbs Sampling
n_iterations = 5000
mu_samples = np.zeros(n_iterations)

# Hyperparameters
prior_mean = 0.0
prior_var = 1.0  # Variance of the prior (standard normal)
data_var = sigma ** 2  # Known variance of the data

# Pre-computed values
data_mean = np.mean(data)

# Compute posterior parameters
posterior_var = 1 / (1 / prior_var + n / data_var)
posterior_mean = posterior_var * (prior_mean / prior_var + n * data_mean / data_var)
# Gibbs sampling iterations
for i in range(n_iterations):
    # Sample from the posterior
    mu_samples[i] = np.random.normal(posterior_mean, np.sqrt(posterior_var))

# Posterior summary
posterior_mean_est = np.mean(mu_samples)
credible_interval = np.percentile(mu_samples, [2.5, 97.5])

print(f"Posterior mean: {posterior_mean_est:.2f}")
print(f"95% credible interval: {credible_interval}")

# Plotting the results
plt.figure(figsize=(10, 6))

# Trace plot
plt.subplot(2, 1, 1)
plt.plot(mu_samples, lw=0.5)
plt.title("Trace Plot")
plt.ylabel("mu")

# Posterior distribution
plt.subplot(2, 1, 2)
plt.hist(mu_samples, bins=30, density=True, alpha=0.7, label="Posterior samples")
plt.axvline(posterior_mean_est, color="red", linestyle="--", label="Posterior mean")
plt.title("Posterior Distribution")
plt.xlabel("mu")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()
