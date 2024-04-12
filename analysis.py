import numpy as np
import pandas as pd
from scipy.stats import normaltest, kurtosis, skew, skewtest, kurtosistest, levy_stable, norm
import matplotlib.pyplot as plt
import torch

def print_statistics(log_returns,
                     ref_mean=None, ref_std=None, ref_skew=None, ref_kurtosis=None,
                     log_returns2=None, n_bins=100,
                     label1='S&P 500', label2='Generated'):
    if log_returns.ndim == 1:
        log_returns = log_returns.reshape(1, -1)
        plot=False
    else:
        plot=True
    means = [r.mean()*252 for r in log_returns]
    stds = [r.std()*np.sqrt(252) for r in log_returns]
    skews = [skew(r) for r in log_returns]
    kurtosises = [kurtosis(r) for r in log_returns]

    n_cols = 1 if log_returns2 is None else 2
    if n_cols == 2:
        means2 = [r.mean()*252 for r in log_returns2]
        stds2 = [r.std()*np.sqrt(252) for r in log_returns2]
        skews2 = [skew(r) for r in log_returns2]
        kurtosises2 = [kurtosis(r) for r in log_returns2]

    print(f'Mean annualised log return (252): {np.mean(means)}')
    if log_returns2 is not None: print(f'Mean of second dataset: {np.mean(means2)}')
    print(f'Std annualised log return (252): {np.mean(stds)}')
    if log_returns2 is not None: print(f'Std of second dataset: {np.mean(stds2)}')
    print(f'Min log return: {log_returns.min()} / Max log return: {log_returns.max()}')
    if log_returns2 is not None: print(f'Min of second dataset: {log_returns2.min()} / Max of second dataset: {log_returns2.max()}')
    print(f'Skewness: {np.mean(skews)} (0=normal)')
    if log_returns2 is not None: print(f'Skewness of second dataset: {np.mean(skews2)}')
    print(f'Kurtosis: {np.mean(kurtosises)} (0=normal under Fisher\'s definition)')
    if log_returns2 is not None: print(f'Kurtosis of second dataset: {np.mean(kurtosises2)}')
    print(f'Normal test: {normaltest(log_returns.flatten())}')
    if log_returns2 is not None: print(f'Normal test of second dataset: {normaltest(log_returns2.flatten())}')
    print(f'Skewness test: {skewtest(log_returns.flatten())}')
    if log_returns2 is not None: print(f'Skewness test of second dataset: {skewtest(log_returns2.flatten())}')
    print(f'Kurtosis test: {kurtosistest(log_returns.flatten())}')
    if log_returns2 is not None: print(f'Kurtosis test of second dataset: {kurtosistest(log_returns2.flatten())}')

    if plot:
        plt.hist(means, bins=n_bins, label=label1, alpha=1.0/n_cols,)
        if log_returns2 is not None: plt.hist(means2, bins=n_bins, label=label2, alpha=1.0/n_cols)
        if ref_mean is not None:
            plt.axvline(ref_mean, color='r', label=label1)
        plt.legend()
        plt.show()

        plt.hist(stds, bins=n_bins, label=label1, alpha=1.0/n_cols)
        if log_returns2 is not None: plt.hist(stds2, bins=n_bins, label=label2, alpha=1.0/n_cols)
        if ref_std is not None:
            plt.axvline(ref_std, color='r', label=label1)
        plt.legend()
        plt.show()

        plt.hist(skews, bins=n_bins, label=label1, alpha=1.0/n_cols)
        if log_returns2 is not None: plt.hist(skews2, bins=n_bins, label=label2, alpha=1.0/n_cols)
        if ref_skew is not None:
            plt.axvline(ref_skew, color='r', label=label1)
        plt.legend()
        plt.show()

        plt.hist(kurtosises, bins=n_bins, label=label1, alpha=1.0/n_cols)
        if log_returns2 is not None: plt.hist(kurtosises2, bins=n_bins, label=label2, alpha=1.0/n_cols)
        if ref_kurtosis is not None:
            plt.axvline(ref_kurtosis, color='r', label=label1)
        plt.legend()
        plt.show()
    else:
        return means[0], stds[0], skews[0], kurtosises[0]

def plot_return_hist_and_dist(log_returns, label, alpha=None, beta=None, second_returns=None, second_label=None, bins=1000):
    _, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(log_returns.flatten(), bins=bins, density=True, alpha=0.5, label=label);

    if second_returns is not None:
        ax.hist(second_returns.flatten(), bins=1000, density=True, alpha=0.5, label=second_label);

    if alpha is not None and beta is not None:
        # Plotting Levy Stable distribution
        xmin,xmax = ax.get_xlim()
        x_axis = np.linspace(xmin, xmax, 1001)
        # Calculating mean and standard deviation
        mean = log_returns.mean()
        std = log_returns.std()
        ax.plot(x_axis, levy_stable.pdf(x_axis, alpha, beta, mean, np.sqrt(0.5)*std), alpha=0.8, label=f'α={alpha} β={beta} Levy Stable');
        # ax.plot(x_axis, norm.pdf(x_axis, mean, std), alpha=0.8, label=f'Normal');
    plt.legend()
    plt.show()

def plot_acf_bars(log_returns, n_lags):
    ret_lags = []
    abs_lags = []
    for i in range(n_lags):
        ret_lags.append([pd.Series(log_returns[k,:]).autocorr(lag=i+1) for k in range(len(log_returns))])
        abs_lags.append([pd.Series(np.abs(log_returns[k,:])).autocorr(lag=i+1) for k in range(len(log_returns))])

    _, ax = plt.subplots(1, 2, figsize=(15, 5))
    means = np.mean(ret_lags, axis=1)
    mad = np.abs(ret_lags - means[:, np.newaxis]).mean(axis=1)
    mins = np.min(ret_lags, axis=1)
    maxes = np.max(ret_lags, axis=1)
    ax[0].errorbar(np.arange(n_lags), means, mad, fmt='ok', lw=3)
    ax[0].errorbar(np.arange(n_lags), means, [means - mins, maxes - means],
                fmt='.k', ecolor='gray', lw=1)
    means = np.mean(abs_lags, axis=1)
    std = np.std(abs_lags, axis=1)
    mins = np.min(abs_lags, axis=1)
    maxes = np.max(abs_lags, axis=1)
    ax[1].errorbar(np.arange(n_lags), means, mad, fmt='ob', lw=3)
    ax[1].errorbar(np.arange(n_lags), means, [means - mins, maxes - means],
                fmt='.b', ecolor='gray', lw=1)