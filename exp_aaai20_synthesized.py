import sys
import numpy as np
from time import sleep
from timeit import default_timer as timer
from SLSE import SLSE
from utils.link_funcs import PolynLink, LogisticLink, LogexpLink
from utils.data_gen import gen_slcnr_obj

# n_samples_range = [50000, 100000, 150000, 200000, 250000, 300000]
n_samples_range = [100000, 200000, 300000, 400000, 500000]
n_features_range = [5 * i for i in range(2, 8)]
n_sub_sample_frac_range = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
n_links_range = [5, 10, 15]
# n_links_range = [20, 30, 40]

link_types = {
    "monomial": lambda k: [PolynLink(degree=3) for _ in range(k)],
    "polynomial": lambda k: [PolynLink(degree=1) for _ in range(k // 3)] +
                            [PolynLink(degree=3) for _ in range(k // 3)] +
                            [PolynLink(degree=5) for _ in range(k - 2 * (k // 3))],
    "logistic": lambda k: [LogisticLink() for _ in range(k)],
    "logexp": lambda k: [LogexpLink() for _ in range(k)],
    "mixed": lambda p, lc, lp: [PolynLink(degree=3) for _ in range(p)] +
                               [LogisticLink() for _ in range(lc)] +
                               [LogexpLink() for _ in range(lp)]
}

data_types = {
    "gaussian": lambda n, p, cov: np.random.multivariate_normal(mean=np.zeros(p),
                                                                cov=cov,
                                                                size=n),
    "sub-gaussian": lambda n, p, std: np.random.uniform(low=-std, high=std,
                                                        size=(n, p))
}
data_covs = {
    'gaussian': lambda p: np.identity(p) / p,
    # 'gaussian': lambda p: np.identity(p),
    # 'sub-gaussian': lambda p: 1 / p
    'sub-gaussian': lambda p: 1 / p
}

date = "20190830"
link_func_type = 'mixed'
data_distr = 'sub-gaussian'

# number of different types of link function
n_poly_links_frac, n_logistic_links_frac, n_logexp_links_frac = 0.3, 0.3, 0.4
n_poly_links, n_logistic_links, n_logexp_links = None, None, None


def gen_link_funcs(k):
    global n_poly_links_frac, n_logistic_links_frac, n_logexp_links_frac
    global n_poly_links, n_logistic_links, n_logexp_links
    if link_func_type != 'mixed':
        return link_types[link_func_type](k)
    else:
        # n_poly_links = int(k * n_poly_links_frac)
        n_poly_links = k // 3
        # n_logistic_links = int(k * n_logistic_links_frac)
        n_logistic_links = k // 3
        n_logexp_links = k - n_poly_links - n_logistic_links
        return link_types['mixed'](n_poly_links, n_logistic_links, n_logexp_links)


# results for varying n
n_features, n_sub_sample_frac = 20, 1
results_vary_n = {
    'n_links': np.array(n_links_range),
    'err-mean': None,
    'err-median': None,
    'err-std': None,
    'err-min': None,
    'n_features': n_features,
    'n_sub_sample_frac': n_sub_sample_frac,
    'n_samples': np.array(n_samples_range),
}
n_repeats = 10
err_rates = np.zeros((n_repeats, len(n_links_range), len(n_samples_range)))
print("===\nTry different n with p={}, S/n={}".format(n_features, n_sub_sample_frac))
for r in range(n_repeats):
    print("===\nRepeat {}".format(r))

    for i, n_links in enumerate(n_links_range):
        F = gen_link_funcs(n_links)
        true_beta = np.random.normal(1, 4, size=(n_links, n_features))

        print("\n--- {} {} link functions".format(n_links, link_func_type))
        if link_func_type == 'mixed':
            print("--- ({} polynomial links, {} logistic links, {} logexp links)"
                  .format(n_poly_links, n_logistic_links, n_logexp_links))
        for j, n_samples in enumerate(n_samples_range):
            start = timer()
            # generate training data
            X_cov = data_covs[data_distr](n_features)
            X = data_types[data_distr](n_samples, n_features, X_cov)
            Z = np.random.randn(n_samples, n_links)

            Y = gen_slcnr_obj(X, beta=true_beta, F=F, Z=Z, eps_std=1)

            model = SLSE(cov_option=2, sub_sample_frac=n_sub_sample_frac)
            model.fit(X=X, Y=Y, Z=Z, F=F)

            err_rates[r, i, j] = model.loss(groundtruth=true_beta,
                                            relative=True).max()
            end = timer()
            print("------ {0} samples from {1} distribution: err = {2:.4f}, taking time={3:.2f}s"
                  .format(n_samples, data_distr, err_rates[r, i, j], end-start))

results_vary_n['err-mean'] = err_rates.mean(axis=0)
results_vary_n['err-median'] = np.median(err_rates, axis=0)
results_vary_n['err-min'] = np.min(err_rates, axis=0)
results_vary_n['err-std'] = err_rates.std(axis=0)
print("mean error rates=\n{}\nmedian error rates={}\nmin error rates={}"
      .format(results_vary_n['err-mean'], results_vary_n['err-median'], results_vary_n['err-min']))


# results for varying p
n_samples, n_sub_sample_frac = 500000, 1.0
results_vary_p = {
    'n_links': np.array(n_links_range),
    'err-mean': None,
    'err-min': None,
    'err-median': None,
    'err-std': None,
    'n_features': np.array(n_features_range),
    'n_sub_sample_frac': n_sub_sample_frac,
    'n_samples': n_samples,
}
n_repeats = 0
err_rates = np.zeros((n_repeats, len(n_links_range), len(n_features_range)))
print("===\nTry different p with n={}, S/n={}".format(n_samples, n_sub_sample_frac))
for r in range(n_repeats):
    print("===\nRepeat {}".format(r))

    for i, n_links in enumerate(n_links_range):
        F = gen_link_funcs(n_links)
        print("\n--- {} {} link functions".format(n_links, link_func_type))
        if link_func_type == 'mixed':
            print("--- ({} polynomial links, {} logistic links, {} logexp links)"
                  .format(n_poly_links, n_logistic_links, n_logexp_links))
        for j, n_features in enumerate(n_features_range):
            start = timer()
            # generate training data
            true_beta = np.random.normal(1, 4, size=(n_links, n_features))
            X_cov = data_covs[data_distr](n_features)
            X = data_types[data_distr](n_samples, n_features, X_cov)
            Z = np.random.randn(n_samples, n_links)

            Y = gen_slcnr_obj(X, beta=true_beta, F=F, Z=Z, eps_std=1)

            model = SLSE(cov_option=2, sub_sample_frac=n_sub_sample_frac)
            model.fit(X=X, Y=Y, Z=Z, F=F)

            err_rates[r, i, j] = model.loss(groundtruth=true_beta,
                                            relative=True).max()
            end = timer()
            print("------ {0} features with X~{1}: err = {2:.4f}, taking time={3:.2f}s"
                  .format(n_features, data_distr, err_rates[r, i, j], end-start))

if n_repeats > 0:
    results_vary_p['err-mean'] = err_rates.mean(axis=0)
    results_vary_p['err-median'] = np.median(err_rates, axis=0)
    results_vary_p['err-min'] = np.min(err_rates, axis=0)
    results_vary_p['err-std'] = err_rates.std(axis=0)
    print("mean error rates=\n{}\nmedian error rates={}\nmin error rates={}"
          .format(results_vary_p['err-mean'], results_vary_p['err-median'], results_vary_p['err-min']))


# results for varying s
n_features, n_samples = 20, 500000
results_vary_s = {
    'n_links': np.array(n_links_range),
    'err-mean': None,
    'err-min': None,
    'err-median': None,
    'err-std': None,
    'n_features': n_features,
    'n_sub_sample_frac': n_sub_sample_frac_range,
    'n_samples': n_samples,
}
n_repeats = 10
err_rates = np.zeros((n_repeats, len(n_links_range), len(n_sub_sample_frac_range)))
print("===\nTry different s with p={}, n={}".format(n_features, n_samples))
for r in range(n_repeats):
    print("===\nRepeat {}".format(r))

    for i, n_links in enumerate(n_links_range):
        F = gen_link_funcs(n_links)
        true_beta = np.random.normal(1, 4, size=(n_links, n_features))
        print("\n--- {} {} link functions".format(n_links, link_func_type))
        if link_func_type == 'mixed':
            print("--- ({} polynomial links, {} logistic links, {} logexp links)"
                  .format(n_poly_links, n_logistic_links, n_logexp_links))
        for j, n_sub_sample_frac in enumerate(n_sub_sample_frac_range):
            start = timer()
            # generate training data
            X_cov = data_covs[data_distr](n_features)
            X = data_types[data_distr](n_samples, n_features, X_cov)
            Z = np.random.randn(n_samples, n_links)

            Y = gen_slcnr_obj(X, beta=true_beta, F=F, Z=Z, eps_std=1)

            model = SLSE(cov_option=2, sub_sample_frac=n_sub_sample_frac)
            model.fit(X=X, Y=Y, Z=Z, F=F)

            err_rates[r, i, j] = model.loss(groundtruth=true_beta,
                                            relative=True).max()
            end = timer()
            print("------ {0} sub-sample-fraction with X~{1}: err = {2:.4f}, taking time={3:.2f}s"
                  .format(n_sub_sample_frac, data_distr, err_rates[r, i, j], end-start))

results_vary_s['err-mean'] = err_rates.mean(axis=0)
results_vary_s['err-min'] = np.min(err_rates, axis=0)
results_vary_s['err-median'] = np.median(err_rates, axis=0)
results_vary_s['err-std'] = err_rates.std(axis=0)
print("mean error rates=\n{}\nmedian error rates={}\nmin error rates={}"
      .format(results_vary_s['err-mean'], results_vary_s['err-median'], results_vary_s['err-min']))

results = {
    'varying_n': results_vary_n,
    'varying_p': results_vary_p,
    'varying_s': results_vary_s
}
if link_func_type == 'mixed':
    filename = "results_for_mixed_link_{}-{}-{}_and_{}_data_{}"\
        .format(int(n_poly_links_frac * 10), int(n_logistic_links_frac * 10),
                int(n_logexp_links_frac * 10), data_distr, date)
else:
    filename = "results_for_{}_link_and_{}_data_{}".format(link_func_type, data_distr, date)
np.savez(filename, **results)

sys.exit(0)
