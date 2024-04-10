import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import json
import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
from scipy.linalg import expm
import tables
import torch
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import iisignature as iisig
import arch
from arch.univariate import Normal, StudentsT, SkewStudent, GeneralizedError
import gym
from gym import spaces
from stable_baselines3 import PPO, TD3, SAC, A2C, DDPG
from gaussianize import Gaussianize
import gan

from torch.utils.tensorboard import SummaryWriter

def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))

def get_time_vector(df, difference=False):
    '''
    Get time vector from dataframe index
    Time vector is the number of calendar years from the first date in the index with the first value being 0
    If difference is True, the time vector is the difference in calendar years between consecutive dates
    '''
    t = np.zeros(len(df), dtype=np.float32)
    t[1:] = (df.index.to_series().diff()[1:].dt.days / 365).values.cumsum()
    if difference:
        t = np.diff(t)
    return t

def get_hparams(path):
    events_files = []
    for file in os.listdir(path):
        if file.startswith('events'):
            events_files.append(file)
    events_files.sort()
    events_name = events_files[0]
    hparam_type = {}
    event_acc = EventAccumulator(path + events_name)
    event_acc.Reload()
    tags = event_acc.Tags()["tensors"]
    for tag in tags:
        name = tag.split('/')[0]
        if name.endswith('params'):
            event_list = event_acc.Tensors(tag)
            param_str = str(event_list[0].tensor_proto.string_val[0])
            param_str = param_str.replace('\\n', '')
            param_str = param_str.replace('\\t', '')
            param_str = param_str.replace('\'', '')
            param_str = param_str.replace('\\', '')
            param_str = param_str.replace('b{', '{')
            if param_str.startswith('{'):
                params = json.loads(param_str)
                hparam_type[name] = params
    return hparam_type

def get_generator(params, path):
    gen_type = params['gen_type']
    z_dims_t = params['z_dims_t'] if  'z_dims_t' in params else 4
    Dx = params['Dx']
    g_state_size = params['g_state_size']
    activation = params['activation']
    seq_dim = Dx - 1
    sample_len = params['sample_len']
    hist_len = params['hist_len']
    log_series = params['log_series'] if 'log_series' in params else True
    if gen_type == "lstmd":
        generator = gan.GenLSTMd(z_dims_t, seq_dim, sample_len, hist_len, hidden_size=g_state_size)
    else:
        raise ValueError('Generator type not recognised')
    generator.load_weights(path)
    return generator

def start_writer(data_params, model_params, train_params, rl_params, env_params):
    '''
    Starts a tensorboard writer and logs data, model and training parameters
    Returns the writer
    '''
    algo = rl_params['algo']
    reg_penalty = train_params['reg_penalty']
    sinkhorn_eps = train_params['sinkhorn_eps']
    seed = env_params['seed']

    if reg_penalty.is_integer() and sinkhorn_eps.is_integer():
        suffix = f"e{int(sinkhorn_eps):d}r{int(reg_penalty):d}s{seed:d}"
    elif reg_penalty.is_integer() and not sinkhorn_eps.is_integer():
        suffix = f"e{sinkhorn_eps:.3g}r{int(reg_penalty):d}s{seed:d}"
    elif not reg_penalty.is_integer() and sinkhorn_eps.is_integer():
        suffix = f"e{int(sinkhorn_eps):d}r{reg_penalty:.3g}s{seed:d}"
    else:
        suffix = f"e{sinkhorn_eps:.3g}r{reg_penalty:.3g}s{seed:d}"

    writer = SummaryWriter(comment=f'_{algo}_{suffix}')
    print(rl_params)
    writer.add_text('RL parameters', pretty_json(rl_params))
    writer.add_text('Env parameters', pretty_json(env_params))
    writer.add_text('Data parameters', pretty_json(data_params))
    writer.add_text('Model parameters', pretty_json(model_params))
    writer.add_text('Training parameters', pretty_json(train_params))
    writer.flush()
    return writer

def get_rl_env(generator, writer, env_params, eval=False, seed=None):
    rate = env_params['interest_rate']
    transaction_cost = env_params['transaction_cost']
    cal_name = env_params['trading_calendar']
    gen_batch_size = env_params['gen_batch_size']
    window_len = env_params['window_len']
    hist_len = env_params['hist_len']
    n_periods = env_params['n_periods']
    max_long = env_params['max_long']
    max_short = env_params['max_short']
    signature = env_params['signature_features']
    n_levels = env_params['rl_n_levels'] if signature else None
    lead_lag = env_params['rl_lead_lag'] if signature else None

    random_dates = env_params['random_dates']
    start_date = None if random_dates else env_params['gen_start_date']
    end_date = None if random_dates else env_params['gen_end_date']

    env = ksig_mmd_sim(n_actions=1,
                       window_len=window_len,
                       n_periods=n_periods,
                       max_long=max_long,
                       max_short=max_short,
                       signature=signature,
                       n_levels = n_levels,
                       lead_lag = lead_lag,
                       generator=generator,
                       gen_batch_size=gen_batch_size,
                       gen_start_date=start_date,
                       gen_end_date=end_date,
                       trading_calendar=cal_name,
                       random_dates=random_dates,
                       hist_len=hist_len,
                       r=rate,
                       transaction_cost=transaction_cost,
                       writer=writer,
                       eval=eval,
                       seed=seed)
    return env

def get_real_data_env(path, env_params, writer=None):
    rate = env_params['interest_rate']
    transaction_cost = env_params['transaction_cost']
    stride = env_params['stride']
    window_len = env_params['window_len']
    n_periods = env_params['n_periods']
    signature = env_params['signature_features']
    n_levels = env_params['rl_n_levels'] if signature else None
    lead_lag = env_params['rl_lead_lag'] if signature else None

    env = ksig_mmd_sim(n_actions=1,
                       window_len=window_len,
                       n_periods=n_periods,
                       real_data=path,
                       stride=stride,
                       r=rate,
                       transaction_cost=transaction_cost,
                       signature=signature,
                       n_levels = n_levels,
                       lead_lag = lead_lag,
                       writer=writer,
                       eval=True)
    return env

def get_rl_agent(RL_algo, env, arguments, tb_path, seed=None):
    if RL_algo == 'PPO': agent = PPO('MlpPolicy', env, verbose=0, tensorboard_log=tb_path, seed=seed, **arguments)
    if RL_algo == 'TD3': agent = TD3('MlpPolicy', env, verbose=0, tensorboard_log=tb_path, seed=seed, **arguments)
    if RL_algo == 'SAC': agent = SAC('MlpPolicy', env, verbose=0, tensorboard_log=tb_path, seed=seed, **arguments)
    if RL_algo == 'A2C': agent = A2C('MlpPolicy', env, verbose=0, tensorboard_log=tb_path, seed=seed, **arguments)
    if RL_algo == 'DDPG': agent = DDPG('MlpPolicy', env, verbose=0, tensorboard_log=tb_path, seed=seed, **arguments)
    return agent

def np_lead_lag_transform(data: np.ndarray, t: np.ndarray, lead_lag: Union[int,list[int]]=1):
    '''
    Transform data to lead-lag format
    data is of shape (seq_len, seq_dim)
    '''
    assert len(data.shape) == 2, 'data must be of shape (seq_len, seq_dim)'
    assert data.shape[0] == t.shape[0], 'data and df_index must have the same length'
    if isinstance(lead_lag, int):
        if lead_lag <= 0: raise ValueError('lead_lag must be a positive integer')
    else:
        for lag in lead_lag:
            if lag <= 0: raise ValueError('lead_lag must be a positive integer')

    # get shape of output
    seq_len = data.shape[0]
    seq_dim = data.shape[1]
    shape = list(data.shape)
    if isinstance(lead_lag, int):
        lead_lag = [lead_lag]
    max_lag = max(lead_lag)
    shape[0] = shape[0] + max_lag
    shape[1] = (len(lead_lag) + 1) * seq_dim

    # create time dimension
    t = np.concatenate([t, np.ones(max_lag) * t[-1]], axis=0).reshape(-1, 1) # pad latter values with last value, shape (seq_len + max_lag, 1)

    # create lead-lag series
    lead_lag_data = np.empty(shape) # shape (seq_len + max_lag, seq_dim * (len(lead_lag) + 1))
    lead_lag_data[:seq_len, :seq_dim] = data # fill in original sequence
    lead_lag_data[seq_len:, :seq_dim] = data[-1] # pad latter values with last value
    for i, lag in enumerate(lead_lag):
        i = i + 1 # skip first seq_dim columns
        lead_lag_data[:lag, i*seq_dim:(i+1)*seq_dim] = 0.0 # pad initial values with zeros
        lead_lag_data[lag:lag+seq_len, i*seq_dim:(i+1)*seq_dim] = data
        lead_lag_data[lag+seq_len-1:, i*seq_dim:(i+1)*seq_dim] = data[-1] # pad latter values with last value
    return np.concatenate([t, lead_lag_data], axis=1)

class GARCH_path_generator():
    '''
    Generate paths based on the following steps:
    1. Create GARCH model from real data by first Gaussianizing the annualised and normalised log returns with the Lambert transform
    2. Generate GARCH residuals using GARCH model build from real data
    3. Feed GARCH residuals and historical path into generator to produce synthetic paths
    Parameters
    ----------
    generator: nn.Module
        Generator model
    df: pd.DataFrame
        Dataframe containing historical data with a datetime index
    start_date: str
        Start date of data used to train GARCH model
    end_date: str
        End date of data used to train GARCH model
    p: int
        Number of lags for GARCH model for squared residuals
    q: int
        Number of lags for GARCH model for variance
    mean_model: str
        Mean model for GARCH model, e.g. 'Constant', 'Zero', 'AR', etc
    vol_model: str
        Volatility model for GARCH model, e.g. 'GARCH', 'ARCH', 'EGARCH', etc
    dist: str
        Distribution for GARCH model, e.g. 'gaussian', 't', 'skewt', 'ged'
    col_idx: int|list[int]
        Column index of data to use for GARCH model
    seed: int
        Random seed for GARCH model
    '''

    def __init__(self, generator: tf.keras.Model, df: pd.DataFrame, start_date: str, end_date: str,
                 p: Optional[int], o: Optional[int], q: Optional[int],
                 mean_model: Optional[str], vol_model: Optional[str], dist: Optional[str],
                 col_idx: Optional[Union[int,list[int]]]=None,
                 seed: Optional[int]=None, random_state: Optional[np.random.RandomState]=None):

        # assert that not both seed and random_state are passed
        assert (seed is None) or (random_state is None), 'only one of seed or random_state can be passed'

        df_gen = df.copy() # df_gen used to generate paths which preserves the original data date range
        df = df.loc[start_date:end_date].copy() # GARCH model trained on data from start_date to end_date

        # Gaussianize data and train GARCH model
        df['log_returns'] = np.log(df.iloc[:, col_idx]).diff()
        df['dt'] = df.index.to_series().diff().dt.days / 365
        df['cal_ann_returns'] = df.loc[:, 'log_returns'] / df.loc[:, 'dt']
        df['norm_cal_ann_returns'] = (df.loc[:, 'cal_ann_returns'] - df.loc[:, 'cal_ann_returns'].mean()) / df.loc[:, 'cal_ann_returns'].std()
        lambert_transform = Gaussianize()
        df.dropna(inplace=True)
        lambert_transform = lambert_transform.fit(df.loc[:, 'norm_cal_ann_returns'])
        df['gaussianized'] = lambert_transform.transform(df.loc[:, 'norm_cal_ann_returns'])
        self.df = df[['gaussianized']].copy() # self.df used to set up GARCH model ending at the start of the item
        garch_model = arch.arch_model(df.loc[:,'gaussianized'], mean=mean_model, vol=vol_model, p=p, o=o, q=q, rescale=True, dist=dist)
        self.res = garch_model.fit(update_freq=0)
        self.garch_params = self.res.params
        print(self.res.summary())

        # prepare data for generator
        self.generator = generator
        df_gen['log_returns'] = np.log(df_gen.iloc[:, col_idx]).diff()
        df_gen['dt'] = df_gen.index.to_series().diff().dt.days / 365
        df_gen['cal_ann_returns'] = df_gen.loc[:, 'log_returns'] / df_gen.loc[:, 'dt']
        df_gen['norm_cal_ann_returns'] = (df_gen.loc[:, 'cal_ann_returns'] - df_gen.loc[:, 'cal_ann_returns'].mean()) / df_gen.loc[:, 'cal_ann_returns'].std()
        df_gen['gaussianized'] = lambert_transform.transform(df_gen.loc[:, 'norm_cal_ann_returns'])
        df_gen.dropna(inplace=True)
        df_gen['log_path'] = np.log(df_gen.iloc[:, col_idx]) - np.log(df_gen.iloc[:, col_idx].iloc[0])
        df_gen['t'] = np.cumsum(df_gen['dt'])
        df_gen['t'] = df_gen['t'] - df_gen['t'].iloc[0]
        self.df_gen = df_gen

        self.max_lag = max(p, o, q)
        self.p = p
        self.o = o
        self.q = q
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.dist = dist
        self.noise_dim = generator.noise_dim
        self.rs = np.random.RandomState(seed) if random_state is None else random_state

    def generate(self, start_date: str, end_date: str, trading_calendar: str, hist_len: int,
                 batch_size: int, n_batches: int=1,
                 device: str='cpu', dtype: torch.dtype=torch.float32):
        '''
        Generate synthetic paths using GARCH model and generator
        The historical portion of the synthetic paths will be drawn from df_gen i.e. real data
        This means all synthetic paths will have the same historical portion
        Parameters
        ----------
        start_date: str
            Start date of synthetic paths
            GARCH model will be conditioned based on data up to this date
        end_date: str
            End date of synthetic paths
        trading_calendar: str
            Trading calendar to use for generating time dimension
        hist_len: int
            Length of historical portion drawn from df_gen
        batch_size: int
            Number of paths to generate
        n_batches: int
            Number of batches to generate
        '''
        # set up timeline
        if pd.to_datetime(end_date) > self.df_gen.index[-1]:
            calendar = mcal.get_calendar(trading_calendar)
            schedule = calendar.schedule(start_date=start_date, end_date=end_date)
            t = np.zeros(len(schedule))
            t[1:] = (schedule.index.to_series().diff()[1:].dt.days / 365).values.cumsum()
        else:
            t = self.df_gen[start_date:end_date]['t'].values
        t = t - t[0]

        # adjust generator sequence length parameter if necessary
        self.sample_len = len(t)
        if self.generator.seq_len != self.sample_len:
            self.generator.seq_len = self.sample_len

        # # prepare time tensor and historical data tensor for generator
        t = np.tile(t, (batch_size, 1))[...,np.newaxis] # shape (batch_size, sample_len, seq_dim)
        x = self.df_gen[start_date:end_date]['log_path'].values
        if len(x) < hist_len:
            raise ValueError('hist_len is longer than the available historical data.')
        x = x - x[0] # rebase historical data to start at 0
        x = np.tile(x, (batch_size, 1))[...,np.newaxis] # shape (batch_size, hist_len, seq_dim)
        x = np.concatenate([t,x], axis=-1)

        if len(self.df_gen.loc[:start_date, 'gaussianized']) < self.max_lag:
            raise ValueError('GARCH initialization period is longer than the available historical data.')

        # create GARCH model with data up to start_date but model parameters will be from self.garch_params trained during init
        garch_model = arch.arch_model(self.df_gen.loc[:start_date, 'gaussianized'],
                                                mean=self.mean_model, vol=self.vol_model,
                                                p=self.p, o=self.o, q=self.q, rescale=False)

        # set distribution of GARCH model with random state passed in the construction for reproducibility
        if self.dist == 'gaussian':
            garch_model.distribution = Normal(seed=self.rs)
        elif self.dist == 't':
            garch_model.distribution = StudentsT(seed=self.rs)
        elif self.dist == 'skewt':
            garch_model.distribution = SkewStudent(seed=self.rs)
        elif self.dist == 'ged':
            garch_model.distribution = GeneralizedError(seed=self.rs)
        else:
            raise ValueError('dist must be gaussian, t, skewt or ged')

        list_path_whist = []
        list_path_wohist = []
        list_log_returns = []

        # for _ in tqdm(range(n_batches)):
        for _ in range(n_batches):
            forecasts = garch_model.forecast(params=self.garch_params, # use GARCH model parameters trained during init
                                            horizon=self.sample_len-1, # forecast sample_len-1 steps
                                            method='simulation',
                                            simulations=self.noise_dim*batch_size)
            noise = forecasts.simulations.residuals[0].T
            noise = noise.reshape(noise.shape[0], self.noise_dim, batch_size).transpose(2, 0, 1) # shape (batch_size, sample_len-1, noise_dim)

            output_whist = self.generator(noise, x)
            output_whist = torch.tensor(output_whist.numpy(), dtype=dtype, device=device, requires_grad=False)
            timeline_whist = output_whist[0, :, 0].numpy()
            output_whist = output_whist[:, :, 1]
            timeline_wohist = timeline_whist[hist_len:]
            if output_whist.ndim == 1:
                output_whist = output_whist.unsqueeze(0)
            path_whist = torch.exp(output_whist)
            output_wohist = output_whist[:, hist_len:]
            output_wohist = output_wohist - output_wohist[:,:1]
            path_wohist = torch.exp(output_wohist)
            log_returns = torch.diff(output_wohist, axis=1)
            list_path_whist.append(path_whist)
            list_path_wohist.append(path_wohist)
            list_log_returns.append(log_returns)

        path_whist = torch.cat(list_path_whist, axis=0).clone().cpu()
        path_wohist = torch.cat(list_path_wohist, axis=0).clone().cpu()
        log_returns = torch.cat(list_log_returns, axis=0).clone().cpu()

        # print(f'path_whist shape: {path_whist.shape}')
        # print(f'path_wohist shape: {path_wohist.shape}')
        # print(f'log_returns shape: {log_returns.shape}')
        # print(f'timeline_whist shape: {timeline_whist.shape}')
        # print(f'timeline_wohist shape: {timeline_wohist.shape}')

        return path_whist, path_wohist, log_returns, timeline_whist, timeline_wohist

def signature_transform(window: np.ndarray, time: np.ndarray, n_levels: int, lead_lag:Optional[list[int]]=None, scale: bool=True) -> np.ndarray:
    assert time.ndim == 1, 'Time must be 1D'
    if window.ndim == 1: window = window.reshape(-1, 1)
    assert window.ndim == 2, 'Window must be 1D or 2D'

    if time[0] != 0: time = time - time[0] # standardise time to start from 0

    window = np.log(window) # log of prices
    window = window - window[0] # standardise to start at 0

    if lead_lag is not None:
        window = np_lead_lag_transform(window, time, lead_lag) # time dimension will be added as well
    else:
        time = time.reshape(-1, 1) # make time 2D for concatenation with window
        window = np.concatenate([time, window], axis=-1)
    # print(window.shape)
    # print(window)

    signature = iisig.sig(window, n_levels)

    # signature terms in a level have factorial decay in magnitude
    # scale signature by factorial of level to bring features to similar scale
    if scale:
        n_seq_terms = window.shape[-1] # sequence dimension
        n_sig_terms = 0
        for i in range(1, n_levels+1):
            new_terms = n_seq_terms**i
            signature[n_sig_terms:n_sig_terms+new_terms] = signature[n_sig_terms:n_sig_terms+new_terms] * math.factorial(i)
            n_sig_terms += new_terms
    # print(signature)
    return signature

def weight_plot(data, step=1, figsize=(6.4,4.8), ax=None, alpha=1.0):
    '''
    Plot series of weights from RL agent episodes
    Each row is the average of the per step weights in the episode
    Plot is a stacked bar plot
    Inputs:
        data: numpy array
            shape of (# episodes, # assets)
        step: int
            drop data points by skipping using step as spacing variable if > 1
    '''
    if step > 1: data = data[::step]
    n_rows, n_cols = data.shape

    _, ax = plt.subplots(figsize=figsize) if ax is None else (None, ax)

    y_offset_pos = np.zeros(n_rows)
    y_offset_neg = np.zeros(n_rows)
    for i in range(n_cols):
        bool_pos = data[:,i] >= 0
        y_offset = bool_pos * y_offset_pos + (1-bool_pos) * y_offset_neg
        ax.bar(np.arange(len(data)), data[:,i], width=1.0, bottom=y_offset, label=('Asset ' + str(i)), alpha=alpha)
        y_offset_pos += bool_pos * data[:,i]
        y_offset_neg += (1-bool_pos) * data[:,i]
    ax.legend(loc='upper left')

def wealth_plot(data, step=1, figsize=(6.4,4.8), log_scale=False, plot_range=True, baseline=None, baseline_range=False, data_label='Agent wealth', baseline_label='Baseline wealth'):
    '''
    Plot wealth path by averaging wealth of a time step across episodes
    Inputs:
        data: numpy array
            shape of (# episodes, # assets)
        step: int
            drop data points by skipping using step as spacing variable if > 1
        log_scale: boolean
            whether to plot in log scale (base 10)
        plot_range: boolean
            whether to plot the min and max of all wealth paths for that period
        baseline: numpy array
            additional plot of a baseline following the same step i.e. should be same size as data
        data_label: string
            plot label for data
        baseline_label: string
            plot label for baseline
    '''
    end_col = data.shape[1]
    if step > 1: data = data[::step]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(stop=end_col, step=step)
    y = np.mean(data, axis=0)
    ax.plot(x, y, label=data_label)

    if plot_range:
        y_min = np.min(data, axis=0)
        y_max = np.max(data, axis=0)
        ax.fill_between(x, y_min, y_max, alpha=0.2)

    if not baseline is None:
        if step > 1: baseline = baseline[::step]
        ax.plot(x, np.mean(baseline, axis=0), label=baseline_label)
        if plot_range:
            y_min = np.min(baseline, axis=0)
            y_max = np.max(baseline, axis=0)
            ax.fill_between(x, y_min, y_max, alpha=0.2)

    if log_scale: ax.set_yscale('log')
    ax.legend(loc='upper left')

def expected_time_growth_rate(weights, μ, Σ, r):
    '''
    Calculates log growth rate of a fixed weight constant rebalanced portfolio
    assuming all risky assets follow correlated GBM i.e. E[log(W(T))]
    Inputs:
        weights: numpy array
            first position is weight for risk free asset
        μ: int
            drop data points by skipping using step as spacing variable if > 1
        Σ: numpy array (n_assets, n_assets)
            covariance matrix
        r: float
            risk free rate
    '''
    w = weights[1:] # risk-free asset weight is position 0
    return (1-w.sum()) * r + (w * μ).sum() - 0.5 * w @ Σ @ w.T

def expected_ensemble_growth_rate(weights, μ, Σ, r):
    '''
    Calculates growth rate of a fixed weight constant rebalanced portfolio
    assuming all risky assets follow correlated GBM i.e. log(E[W(T)])
    Inputs:
        weights: numpy array
            first position is weight for risk free asset
        μ: int
            drop data points by skipping using step as spacing variable if > 1
        Σ: numpy array (n_assets, n_assets)
            covariance matrix
        r: float
            risk free rate
    '''
    w = weights[1:] # risk-free asset weight is position 0
    return (1-w.sum()) * r + (w * μ).sum()

def mean_avg_deviation(data, axis=-1):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)

def variance_log_growth_rate(weights, Σ):
    '''
    Calculates variance of log growth rate of a fixed weight constant rebalanced portfolio
    assuming all risky assets follow correlated GBM i.e. Var[log(W(T))]
    Inputs:
        weights: numpy array
            first position is weight for risk free asset
        μ: int
            drop data points by skipping using step as spacing variable if > 1
        Σ: numpy array (n_assets, n_assets)
            covariance matrix
        r: float
            risk free rate
    '''
    w = weights[1:] # risk-free asset weight is position 0
    return (w @ Σ @ w.T).sum()

def process_action(action):
    '''
    Input is action which are weights for risky assets that may not sum to one
    Determine the weight for the risk-free assets so that total weight sums to one
    Return weights of all asset where risk-free asset is the first element
    '''
    n_assets = len(action)
    weights = np.zeros((n_assets + 1))
    weights[0] = 1. - action.sum()
    weights[1:] = action
    return weights

class pm_env():
    '''
    Portfolio Management Environment
    Inputs:
    Reward is based on log return of weights generated from action
    Reuse trajectories to compute new actions/rewards for the same trajectory after agent has been trained on the trajectory
    This assumes actions do not affect the outcome of the trajectory

    Parameters:
    n_actions: int
        number of risky assets
    window_len: int
        length of window for observation
    n_periods: int
        number of periods where action can be taken
    baseline_weights: numpy array
        weights for baseline portfolio
    verbose: bool
        whether to print out episode results
    writer: str or torch.utils.tensorboard.SummaryWriter
        writer for tensorboard
    real_data: str
        path to real data csv file
    stride: int
        number of periods to skip after each episode for real data
    generator: GARCH_path_generator
        generator for synthetic data
    gen_batch_size: int
        number of generated paths in for each batch the generator outputs
    hist_len: int
        length of history for each generated path which is extracted from historical data and used at the start of each generated path
    gen_start_date: str
        start date for generating synthetic data which includes historical data
    gen_end_date: str
        end date for generating synthetic data which includes historical data
    trading_calendar: str
        trading calendar for generating synthetic data which is used if the gen_end_date is beyond the last date in the historical data
    signature: bool
        whether to use signature feature transformation for observation
    lead_lag: list[int]
        list of lead and lag values for signature feature transformation
    r: float
        risk free rate which is a constant
    seed: int
        random seed
    dtype: torch.dtype
        data type for torch tensors
    device: torch.device
        device for torch tensors
    plot_episode_freq: int
        frequency of plotting episode results (plot_episode)
    eval: bool
        whether environment is for evaluation
    '''

    # Define constants for clearer code
    INITIAL_WEALTH = 1.
    MIN_POS = -1.
    MAX_POS = 2.

    def __init__(self, n_actions: int, window_len: int, n_periods: int,
                 baseline_weights: Optional[Union[np.ndarray,List]]=None,
                 verbose: bool=True, writer: Optional[Union[str,torch.utils.tensorboard.SummaryWriter]]=None,
                 real_data: Optional[str]=None, stride: Optional[int]=None,
                 generator: GARCH_path_generator=None, gen_batch_size: Optional[int]=None, hist_len: Optional[int]=None,
                 gen_start_date:Optional[str]=None, gen_end_date:Optional[str]=None, trading_calendar: Optional[str]=None, random_dates: bool=False,
                 signature: bool=False, n_levels: Optional[int]=None, lead_lag: Optional[List[int]]=None,
                 r: Optional[float]=None, transaction_cost: Optional[float]=0.,
                 seed: Optional[int]=None,
                 dtype: Optional[torch.dtype]=torch.float32, device: Optional[torch.device]=torch.device('cpu'),
                 plot_episode_freq: int=100, eval: bool=False):

        assert not (generator is None and real_data is None), 'Must have either generator or real_data'
        assert not (generator is not None and real_data is not None), 'Cannot have both generator and real_data'
        self.signature = signature
        if signature:
            assert n_levels is not None, 'Must provide number of levels for signature feature transformation'
            self.n_levels = n_levels
            self.lead_lag = lead_lag

        self.r = r
        self.transaction_cost = transaction_cost
        self.n_actions = n_actions
        self.n_assets = self.n_actions + 1
        if baseline_weights is None:
            self.baseline_weights = np.ones((self.n_assets)) / self.n_actions
            self.baseline_weights[0] = 0.
        else:
            self.baseline_weights = np.array(baseline_weights)
            assert self.baseline_weights.shape == (self.n_assets,), 'Baseline weights must have same shape as number of assets'
            assert np.isclose(self.baseline_weights.sum(), 1.), 'Baseline weights must sum to one'
        self.rng = np.random.default_rng(seed=seed)
        self.verbose = verbose

        self.window_len = window_len # length of window for observation
        self.n_periods = n_periods # number of periods where action can be taken
        self.episode_counter = 0 # keep track of number of episodes
        self.steps_counter = 0 # keep track of number of steps
        self.plot_episode_freq = plot_episode_freq
        self.eval = eval

        self.generator = generator
        if generator is not None:
            self.hist_len = hist_len
            self.gen_start_date = gen_start_date
            self.gen_end_date = gen_end_date
            self.trading_calendar = trading_calendar
            self.gen_batch_size = gen_batch_size # number of generated paths in batch
            self.device = device
            self.dtype = dtype
            self.random_dates = random_dates
            if self.random_dates:
                # start date is randomly chosen from the first n of the dates where n = len(gen_dates) - hist_len - n_periods
                # end date is chosen so that there are a total of hist_len + n_periods dates
                self.gen_dates = self.generator.df_gen.index[self.generator.max_lag+1:].to_series() # must leave sufficient data for GARCH(p,q) to generate noise
                start_date_idx = self.rng.integers(0, len(self.gen_dates) - self.hist_len - self.n_periods)
                # print(f'New start date index: {start_date_idx}')
                self.gen_start_date = self.gen_dates[start_date_idx].date().isoformat() # 'yyyy-mm-dd'
                self.gen_end_date = self.gen_dates[start_date_idx + self.hist_len + self.n_periods - 1].date().isoformat()
                # self.gen_batch_size = 1 # only generate one batch at a time if using random dates else all samples in batch will be using the same dates
            gen_output = self.generator.generate(self.gen_start_date, self.gen_end_date, self.trading_calendar, self.hist_len, self.gen_batch_size, 1, device)
            batch, _, _, self.time, _ = gen_output
            # self.batch are paths (non-log) of shape (batch_size, sample_len) sample_len = hist_len + n_periods
            # self.time are the corresponding time in calendar year starting from 0 of shape (batch_size, sample_len)
            if batch.ndim == 2: batch = batch.unsqueeze(-1) # add channel dimension if 1D so that last dimension indicates number of assets
            self.batch = batch.detach().numpy() if batch.device == self.device else batch.detach().numpy().to(self.device)
            self.dts = np.diff(self.time)
            self.elapsed_time = self.time[-1]
            self.ib = 0
            if self.r is None: assert self.batch.shape[-1] == self.n_actions + 1, 'Generator must have n_actions + 1 assets if r is None'
            else: assert self.batch.shape[-1] == self.n_actions, 'Number of actions must be equal to number of assets if r is provided'

        # hd5f database for storing all episodes
        data_filename = 'data_of_eval_episodes.h5' if eval else 'data_of_episodes.h5'
        if real_data is not None: data_filename = 'data_of_real_data_episodes.h5'
        if type(writer) == str:
            self.writer = SummaryWriter(writer)
            self.hd5f_path = writer + f'/{data_filename}'
        elif type(writer) == torch.utils.tensorboard.SummaryWriter:
            self.writer = writer
            self.hd5f_path = writer.log_dir + f'/{data_filename}'
        else:
            self.hd5f_path = f'./{data_filename}'
            self.writer = None
        f = tables.open_file(self.hd5f_path, mode='w')
        atom = tables.Float32Atom()
        f.create_earray(f.root, 'paths_of_episodes', atom, (0, self.window_len + self.n_periods, self.n_actions)) # +1 for initialised prices of 1.0
        f.create_earray(f.root, 'rewards_of_episodes', atom, (0, self.n_periods))
        f.create_earray(f.root, 'weights_of_episodes', atom, (0, self.n_periods, self.n_assets))
        f.create_earray(f.root, 'wealth_of_episodes', atom, (0, self.n_periods + 1)) # +1 for initialised wealth of 1.0
        f.create_earray(f.root, 'baseline_wealth_of_episodes', atom, (0, self.n_periods + 1)) # +1 for initialised wealth of 1.0
        f.close()

        # set up for real data which can be used with reusing trajectories
        if real_data is not None:
            self.start_period = 0
            self.real_df = pd.read_csv(real_data, index_col=0, parse_dates=True)
            self.time = get_time_vector(self.real_df)
            self.dts = np.diff(self.time)
            self.real_data = self.real_df.values # (len(self.real_data), n_actions or n_actions + 1) depending on whether r is provided
            self.stride = stride
            self.elapsed_time = 0
            if r is None: assert self.real_data.shape[-1] == self.n_actions + 1, 'Real data must have n_actions + 1 assets if r is None'
            else: assert self.real_data.shape[-1] == self.n_actions, 'Number of actions must be equal to number of assets if r is provided'
        else:
            self.real_data = None

        # self.debug_mode = True

    def pm_env_reset(self, reset_periods):
        self.episode_rewards = []
        self.episode_wealth = [self.INITIAL_WEALTH]
        self.episode_baseline_wealth = [self.INITIAL_WEALTH]
        self.episode_weights = []
        self.baseline_wealth = self.INITIAL_WEALTH # current wealth of baseline
        self.agent_wealth = self.INITIAL_WEALTH # current wealth of agent
        self.baseline_bankrupt = False

        if self.generator is not None:
            self.episode_times = self.time[self.ib]
            path = self.batch[self.ib] # get next episode from batch
            self.episode_path = path # save episode path for saving
            self.batch_period = 0 # keep track of time index for batch
            self.S = path[self.batch_period, 1:] if self.r is None else path[self.batch_period, :]
            self.B = 1.
            self.curr_step = np.array([self.B] + list(self.S))
            self.dt = self.dts[self.batch_period] # first value of self.dts is for batch_period 0 to 1

            self.ib += 1 # increment batch index to get next episode
            if self.ib == self.gen_batch_size: # generate new batch if reached end of batch
                self.ib = 0
                if self.random_dates:
                    start_date_idx = self.rng.integers(0, len(self.gen_dates) - self.hist_len - self.n_periods)
                    # print(f'New start date index: {start_date_idx}')
                    self.gen_start_date = self.gen_dates[start_date_idx].date().isoformat() # 'yyyy-mm-dd'
                    self.gen_end_date = self.gen_dates[start_date_idx + self.hist_len + self.n_periods - 1].date().isoformat()
                gen_output = self.generator.generate(self.gen_start_date, self.gen_end_date, self.trading_calendar, self.hist_len, self.gen_batch_size, 1, self.device)
                batch, _, _, self.time, _ = gen_output
                # self.batch are paths (non-log) of shape (batch_size, sample_len) sample_len = hist_len + n_periods
                # self.time are the corresponding time in calendar year starting from 0 of shape (batch_size, sample_len)
                if batch.ndim == 2: batch = batch.unsqueeze(-1)
                self.batch = batch.detach().numpy() if batch.device == self.device else batch.detach().numpy().to(self.device)
                self.dts = np.diff(self.time)
                self.elapsed_time = self.time[-1]

            # if self.debug_mode:
            #     # print(f'Batch_period: {self.batch_period} / Curr_date: {self.df_index[self.batch_period].date().isoformat()} / dt: {self.dt}')
            #     temp_gen_dates = self.gen_dates[self.gen_start_date:self.gen_end_date]
            #     print(f'Batch_period: {self.batch_period} / Curr_date: {temp_gen_dates[self.batch_period].date().isoformat()} / dt: {self.dt}')
            #     print(f'Curr step: {self.curr_step}')

        elif self.real_data is not None:
            self.S = np.ones((self.n_actions), dtype=np.float64)
            self.B = 1.
            self.curr_step = np.append(self.B, self.S)
            self.episode_path = [self.S]

            # periods needed for episode is window_len + n_periods NOTE: reset_periods will be window_len - 1
            periods_needed = self.window_len + self.n_periods

            # get data for episode starting from start_period
            self.episode_data = self.real_data[self.start_period:self.start_period + periods_needed]

            # check if sufficient number of periods in episode
            if self.episode_data.shape[0] < periods_needed:
                print(f'Not enough periods in real data for episode. Need {periods_needed} but only have {self.episode_data.shape[0]}')
                self.episode_path = np.empty((periods_needed, self.n_actions))
                self.dt = None
                self.position = [None]
                return
            else:
                # calculate time elapsed from start_period to start_period + periods_needed
                start_date = self.real_df.index[self.start_period]
                end_date = self.real_df.index[self.start_period + periods_needed - 1]
                self.elapsed_time += (end_date - start_date).days / 365
                print(f'Data period from {start_date.date().isoformat()} to {end_date.date().isoformat()} / Index: {self.start_period} to {self.start_period + periods_needed}')

            # normalise to start at 1
            self.episode_data = self.episode_data / self.episode_data[0, :]

            # keep track of time index for episode
            self.real_data_current_period = 0

            # dts needed for episode is window_len + n_periods - 1
            self.episode_dts = self.dts[self.start_period:self.start_period + periods_needed - 1]

            # first value of self.dts is for batch_period 0 to 1
            self.dt = self.episode_dts[self.real_data_current_period]

            # increment start period for next episode
            self.start_period += self.stride

        # keep track of periods simulated after reset simulation note that simulate_one_step does not increment this
        self.current_period = 0 # equals to number of actions taken
        for t in range(reset_periods): self.simulate_one_step()

        self.position = np.zeros((self.n_actions,), dtype=np.float32) # for obs if used

    def simulate_one_step(self):
        '''
        if reusing trajectories, check counter to see if reached end of life for current trajectory
        -> if yes, then simulate as per normal and reset n_reuse_trajectory_counter
        -> if no, then reuse trajectory at traj_step of last episode and increment traj_step
        '''

        if self.generator is not None:
            self.batch_period += 1 # increment batch_period for next time step in batch, first call increments this from 0 to 1
            next_step = self.episode_path[self.batch_period]
            self.S = next_step[1:] if self.r is None else next_step
            self.B *= np.exp(next_step[1] * self.dt) if self.r is None else np.exp(self.r * self.dt) # dt is from prev batch_period to current batch_period
            if self.current_period < self.n_periods: # if not at terminal time
                self.dt = self.dts[self.batch_period] # calculates dt from current batch_period to next batch_period NOTE: first dt assigned in pm_env_reset
            next_step = np.array([self.B] + list(self.S)) # don't update self.curr_step until it is saved in self.prev_step

            # if self.debug_mode:
            #     # print(f'Batch_period: {self.batch_period} / Curr_date: {self.df_index[self.batch_period].date().isoformat()} / dt: {self.dt}')
            #     temp_gen_dates = self.gen_dates.loc[self.gen_start_date:self.gen_end_date]
            #     print(f'Batch_period: {self.batch_period} / Curr_date: {temp_gen_dates[self.batch_period].date().isoformat()} / dt: {self.dt}')
            #     print(f'Curr step: {next_step}')

        elif self.real_data is not None:
            self.real_data_current_period += 1
            if self.r is None:
                next_step = self.episode_data[self.real_data_current_period]
            else:
                self.B = self.B * np.exp(self.r * self.dt)
                self.S = self.episode_data[self.real_data_current_period]
                next_step = np.append(self.B, self.S)
            self.episode_path.append(self.S)
            if self.real_data_current_period < self.n_periods: # if not at terminal time
                self.dt = self.episode_dts[self.real_data_current_period] # calculates dt from current batch_period to next batch_period

        self.prev_step = self.curr_step
        self.curr_step = next_step

    def pm_env_step(self, action):
        self.current_period += 1 # value is 1 after first action i.e. self.n_periods = num of actions
        self.simulate_one_step() # simulate next step prices AFTER action is taken
        simple_return = self.curr_step / self.prev_step # return factor for all assets i.e. 1 + % change in price

        # baseline should have no possibility of bankruptcy (if dt small enough)
        # but can happen when dt not small enough and volatility is high
        if self.baseline_bankrupt:
            self.baseline_wealth = 0.
        else:
            baseline_return = (self.baseline_weights * simple_return).sum()
            if baseline_return <= 0:
                self.baseline_bankrupt = True
                self.baseline_wealth = 0.
            else:
                self.baseline_wealth *= baseline_return
        self.episode_baseline_wealth.append(self.baseline_wealth)

        weights = self.process_action(action) # adds weight for cash where total sums to one
        portfolio_return = (weights * simple_return).sum()

        # deduct transaction cost before wealth is updated to new value based on new stock prices
        if self.transaction_cost > 0:
            prev_weights = self.episode_weights[-1] if len(self.episode_weights) > 0 else np.zeros((self.n_assets,))
            transaction_cost = self.transaction_cost * np.abs(weights[1:] - prev_weights[1:]).sum() * self.agent_wealth
        else:
            transaction_cost = 0.

        if portfolio_return <= 0:                                   # check for bankruptcy
            done = True                                             # if bankrupt then episode is done
            reward = np.log(np.nextafter(0, 1))                     # lowest reward possible based on log of smallest positive number
            self.agent_wealth = 0.                                  # agent bankrupt
        else:
            done = False
            new_wealth = self.agent_wealth * portfolio_return - transaction_cost
            reward = np.log(new_wealth / self.agent_wealth) # log return of wealth
            self.agent_wealth = new_wealth

        self.episode_wealth.append(self.agent_wealth)
        self.episode_weights.append(weights)
        self.episode_rewards.append(reward)

        # if self.debug_mode:
        #     print(f'Curr_period: {self.current_period} / simple_return: {simple_return} / Portfolio_return: {portfolio_return}')
        #     print(f'Wealth from {self.episode_wealth[-2]} to {self.episode_wealth[-1]}')
        #     print(f'Weights from {prev_weights} to {self.episode_weights[-1]}')
        #     print(f'Reward: {reward} / Transaction cost: {transaction_cost}')

        if self.current_period == self.n_periods: done = True # reached terminal time

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        if done: self.pm_env_done()

        self.position = action.astype(np.float32) # for obs if used

        return reward, done, info

    def pm_env_done(self):
        # actions to complete if env is done
        if self.current_period < self.n_periods: # if agent is bankrupt but not at terminal time yet
            self.simulate_baseline_to_terminal_time()
        weights = np.array(self.episode_weights)

        self.episode_counter += 1
        self.steps_counter += len(self.episode_rewards)
        if self.verbose:
            print(f'E{self.episode_counter} / S{self.steps_counter}: ' +
                  f'Baseline Wealth = {self.baseline_wealth:.4f} / Agent Final Wealth = {self.agent_wealth:.4f} / ' +
                  f'Average Weights and Std: {weights.mean(axis=0)} / {np.sqrt(weights.var(axis=0))}')
        self.save_episode()
        if self.episode_counter % self.plot_episode_freq == 0:
            self.plot_episode()

    def save_episode(self):
        f = tables.open_file(self.hd5f_path, mode='a')
        f.root.paths_of_episodes.append(np.array(self.episode_path)[np.newaxis,...])
        f.root.weights_of_episodes.append(np.array(self.episode_weights)[np.newaxis,...])
        f.root.rewards_of_episodes.append(np.array(self.episode_rewards)[np.newaxis,...])
        f.root.wealth_of_episodes.append(np.array(self.episode_wealth)[np.newaxis,...])
        f.root.baseline_wealth_of_episodes.append(np.array(self.episode_baseline_wealth)[np.newaxis,...])
        f.close()

    def baseline_results(self, growth_rates=True):
        f = tables.open_file(self.hd5f_path, mode='r')
        baseline_average_final_wealth = np.average([w[-1] for w in f.root.baseline_wealth_of_episodes])
        baseline_final_wealth_std = np.std([w[-1] for w in f.root.baseline_wealth_of_episodes])
        baseline_final_wealth_mad = mean_avg_deviation([w[-1] for w in f.root.baseline_wealth_of_episodes])
        baseline_wealth_std = np.mean([np.std(w) for w in f.root.baseline_wealth_of_episodes])
        if growth_rates:
            if self.generator is None:
                avg_elapsed_time = self.elapsed_time / self.episode_counter
            else:
                avg_elapsed_time = self.elapsed_time
            num_episodes = len(f.root.baseline_wealth_of_episodes)
            num_bankrupt = len([w[-1] for w in f.root.baseline_wealth_of_episodes if w[-1] <= 0])
            baseline_time_growth_rate = np.average(np.log([w[-1] for w in f.root.baseline_wealth_of_episodes if w[-1] > 0]) / avg_elapsed_time)
            baseline_ensemble_growth_rate = np.average(np.log(baseline_average_final_wealth) / avg_elapsed_time)
        f.close()

        if self.writer is not None:
            text = f'Avg final wealth: {baseline_average_final_wealth:.4f}  '
            text += f'\nFinal wealth std: {baseline_final_wealth_std:.4f}  '
            text += f'\nFinal wealth mad: {baseline_final_wealth_mad:.4f}  '
            text += f'\nAvg wealth path std: {baseline_wealth_std:.4f}  '
            text += f'\nNum bankrupt: {num_bankrupt} / {num_episodes}  '
            if growth_rates: text += f'\nGrowth rate: {baseline_time_growth_rate:.4f}'
            eval_text = ' (eval)' if self.eval else ''
            if self.real_data is not None:
                eval_text = ' (real data eval)'
            self.writer.add_text(f'baseline_wealth_stats{eval_text}', text)
        if growth_rates:
            return baseline_average_final_wealth, baseline_final_wealth_std, baseline_final_wealth_mad, baseline_wealth_std, baseline_time_growth_rate, baseline_ensemble_growth_rate, num_bankrupt, num_episodes
        else:
            return baseline_average_final_wealth, baseline_final_wealth_std, baseline_final_wealth_mad, baseline_wealth_std, num_bankrupt, num_episodes

    def agent_results(self, growth_rates=True):
        f = tables.open_file(self.hd5f_path, mode='r')
        average_final_wealth = np.average([w[-1] for w in f.root.wealth_of_episodes])
        final_wealth_std = np.std([w[-1] for w in f.root.wealth_of_episodes])
        final_wealth_mad = mean_avg_deviation([w[-1] for w in f.root.wealth_of_episodes])
        wealth_std = np.mean([np.std(w) for w in f.root.wealth_of_episodes])
        if growth_rates:
            if self.generator is None:
                avg_elapsed_time = self.elapsed_time / self.episode_counter
            else:
                avg_elapsed_time = self.elapsed_time
            num_episodes = len(f.root.wealth_of_episodes)
            num_bankrupt = len([w[-1] for w in f.root.wealth_of_episodes if w[-1] <= 0])
            time_growth_rate = np.average(np.log([w[-1] for w in f.root.wealth_of_episodes if w[-1] > 0]) / avg_elapsed_time)
            ensemble_growth_rate = np.average(np.log(average_final_wealth) / avg_elapsed_time)
        f.close()

        if self.writer is not None:
            text = f'Avg final wealth: {average_final_wealth:.4f}  '
            text += f'\nFinal wealth std: {final_wealth_std:.4f}  '
            text += f'\nFinal wealth mad: {final_wealth_mad:.4f}  '
            text += f'\nAvg wealth path std: {wealth_std:.4f}  '
            text += f'\nNum bankrupt: {num_bankrupt} / {num_episodes}  '
            if growth_rates: text += f'\nGrowth rate: {time_growth_rate:.4f}  '
            eval_text = ' (eval)' if self.eval else ''
            if self.real_data is not None:
                eval_text = ' (real data eval)'
            self.writer.add_text(f'agent_wealth_stats{eval_text}', text)
        if growth_rates:
            return average_final_wealth, final_wealth_std, final_wealth_mad, wealth_std, time_growth_rate, ensemble_growth_rate, num_bankrupt, num_episodes
        else:
            return average_final_wealth, final_wealth_std, final_wealth_mad, wealth_std, num_bankrupt, num_episodes

    def weight_plot(self, step=1, title='weight_plot'):
        f = tables.open_file(self.hd5f_path, mode='r')
        data = np.array([np.mean(np.array(weights), axis=0) for weights in f.root.weights_of_episodes])
        weight_plot(data, step)
        f.close()
        if self.writer is not None:
            fig = pickle.loads(pickle.dumps(plt.gcf()))
            self.writer.add_figure(title, fig)

    def weight_mae_plot(self, step=1, title='weight_mae_plot'):
        f = tables.open_file(self.hd5f_path, mode='r')
        data = np.array([np.absolute(weights - np.mean(weights, axis=0)).mean(axis=0) for weights in f.root.weights_of_episodes])
        weight_plot(data, step)
        f.close()
        if self.writer is not None:
            fig = pickle.loads(pickle.dumps(plt.gcf()))
            self.writer.add_figure(title, fig)

    def agent_wealth_plot(self, step=1, log_scale=False, plot_range=True, baseline=False, baseline_range=False, title='agent_wealth_plot'):
        f = tables.open_file(self.hd5f_path, mode='r')
        # max_len = np.max([len(a) for a in f.root.wealth_of_episodes])
        # data = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in f.root.wealth_of_episodes])
        data = f.root.wealth_of_episodes
        if baseline:
            # baseline_data = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in f.root.baseline_wealth_of_episodes])
            baseline_data = f.root.baseline_wealth_of_episodes
            wealth_plot(data, step, log_scale=log_scale, plot_range=plot_range, baseline=baseline_data, baseline_range=baseline_range)
        else: wealth_plot(data, step, log_scale=log_scale, plot_range=plot_range)
        f.close()
        if self.writer is not None:
            fig = pickle.loads(pickle.dumps(plt.gcf()))
            self.writer.add_figure(title, fig)

    def agent_log_wealth_plot(self, step=1, plot_range=True, baseline=False, baseline_range=False, title='agent_log_wealth_plot'):
        f = tables.open_file(self.hd5f_path, mode='r')
        # max_len = np.max([len(a) for a in self.wealth_of_episodes])
        # data = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in self.wealth_of_episodes])
        # data = np.log(data)
        data = [episode for episode in f.root.wealth_of_episodes if episode[-1] > 0]
        data = np.log(data)
        if baseline:
            # baseline_data = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in self.baseline_wealth_of_episodes])
            # baseline_data = np.log(baseline_data)
            baseline_data = [episode for episode in f.root.baseline_wealth_of_episodes if episode[-1] > 0]
            baseline_data = np.log(baseline_data)
            wealth_plot(data, step, plot_range=plot_range, baseline=baseline_data, baseline_range=baseline_range)
        else: wealth_plot(data, step, plot_range=plot_range)
        f.close()
        if self.writer is not None:
            fig = pickle.loads(pickle.dumps(plt.gcf()))
            self.writer.add_figure(title, fig)

            fig = pickle.loads(pickle.dumps(plt.gcf()))
            self.writer.add_figure(title, fig)

    def process_action(self, action):
        '''
        Input is action which are weights for risky assets that may not sum to one
        Determine the weight for the risk-free assets so that total weight sums to one
        Return weights of all asset where risk-free asset is the first element
        '''
        weights = np.zeros((self.n_assets))
        weights[0] = 1. - action.sum()
        weights[1:] = action
        return weights

    def simulate_baseline_to_terminal_time(self):
        '''
        Simulate the baseline wealth to maturity when agent is already bankrupt hence episode terminated
        Assumes baseline wealth cannot go bankrupt based on baseline weights
        '''
        for t in range(self.n_periods - self.current_period):
            self.simulate_one_step()
            if self.baseline_bankrupt:
                self.baseline_wealth = 0.
            else:
                simple_return = self.curr_step / self.prev_step               # return factor for all assets i.e. 1 + % change in price
                baseline_return = (self.baseline_weights * simple_return).sum() # baseline should have no possibility of bankruptcy (if dt small enough)
                self.baseline_wealth *= baseline_return
            self.episode_baseline_wealth.append(self.baseline_wealth)
            self.episode_wealth.append(0.)
            self.episode_weights.append(np.zeros((self.n_assets)))
            self.episode_rewards.append(0.)
            self.current_period += 1

    def final_wealth_plot(self):
        f = tables.open_file(self.hd5f_path, mode='r')
        baseline_final_wealths = np.array([w[-1] for w in f.root.baseline_wealth_of_episodes])
        agent_final_wealths = np.array([w[-1] for w in f.root.wealth_of_episodes])
        f.close()

        baseline_binwidth = 1 if baseline_final_wealths.max() > 99 else 0.1
        if baseline_final_wealths.max() > 999: baseline_binwidth = 10
        baseline_bins = int((baseline_final_wealths.max() - baseline_final_wealths.min()) // baseline_binwidth)

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.hist(baseline_final_wealths, bins=baseline_bins)
        plt.title('Baseline final wealth')
        if self.writer is not None:
            fig = pickle.loads(pickle.dumps(plt.gcf()))
            tag = 'Plots/Baseline Eval Final Wealth' if self.eval else 'Plots/Baseline Training Final Wealth'
            self.writer.add_figure(tag, fig)

        binwidth = 0.1 if agent_final_wealths.max() > 9 else 0.01
        if agent_final_wealths.max() > 99: binwidth = 1
        elif agent_final_wealths.max() > 999: binwidth = 10
        bins = int((agent_final_wealths.max() - agent_final_wealths.min()) // binwidth)

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.hist(agent_final_wealths, bins=bins)
        ax.set_xlim([0,int(np.ceil(np.percentile(agent_final_wealths, 99.9)))])
        plt.title('Agent final wealth')

        if self.writer is not None:
            fig = pickle.loads(pickle.dumps(plt.gcf()))
            tag = 'Plots/Agent Eval Final Wealth' if self.eval else 'Plots/Agent Training Final Wealth'
            self.writer.add_figure(tag, fig)

    def plot_episode(self):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.plot(self.episode_path[self.window_len:], label='Path', color='black');
        ax.plot(self.episode_wealth, label='Wealth', color='red');
        weight_plot(np.array(self.episode_weights), step=1, ax=ax, alpha=0.5);
        ax.legend();
        if self.writer is not None:
            fig = pickle.loads(pickle.dumps(plt.gcf()))
            tag = 'Plots/Episode (eval)' if self.eval else 'Plots/Episode'
            if self.real_data is not None:
                tag = 'Plots/Real Data Eval'
            self.writer.add_figure(tag, fig, global_step=self.episode_counter)
            plt.close()

    def seed(self, seed):
        pass

class ksig_mmd_sim(gym.Env, pm_env):
    '''
    Portfolio Management Environment
    Reward is based on log return of weights generated from action
    Observation is based on window length of prices, current wealth, current position and time to next period
    '''
    def __init__(self, n_actions: int, window_len: int, n_periods: int,
                 max_long: Optional[float]=None, max_short: Optional[float]=None,
                 baseline_weights: Optional[Union[np.ndarray,List]]=None,
                 verbose: bool=True, writer: Optional[Union[str,torch.utils.tensorboard.SummaryWriter]]=None,
                 real_data: Optional[str]=None, stride: Optional[int]=None,
                 generator: GARCH_path_generator=None, gen_batch_size: Optional[int]=None, hist_len: Optional[int]=None,
                 gen_start_date:Optional[str]=None, gen_end_date:Optional[str]=None, trading_calendar: Optional[str]=None, random_dates: bool=False,
                 signature: bool=False, n_levels: Optional[int]=None, lead_lag: Optional[List[int]]=None,
                 r: Optional[float]=None, transaction_cost: float=0.,
                 seed: Optional[int]=None,
                 dtype: Optional[torch.dtype]=torch.float32, device: Optional[torch.device]=torch.device('cpu'),
                 plot_episode_freq: int=100, eval: bool=False):

        super().__init__(n_actions, window_len, n_periods,
                         baseline_weights, verbose, writer,
                         real_data, stride,
                         generator, gen_batch_size, hist_len,
                         gen_start_date, gen_end_date, trading_calendar, random_dates,
                         signature, n_levels, lead_lag,
                         r, transaction_cost, seed, dtype, device,
                         plot_episode_freq, eval)

        max_long = self.MAX_POS if max_long is None else max_long
        max_short = self.MIN_POS if max_short is None else max_short
        self.action_space = spaces.Box(low=max_short, high=max_long, shape=(self.n_actions,), dtype=np.float32)
        if signature:
            n_seq_terms = 1 + n_actions + n_actions*len(lead_lag) # time dim + number of risky assets + number of additional terms due to lead-lag transform
            n_sig_terms = 0
            for i in range(1, n_levels+1):
                n_sig_terms += n_seq_terms**i
            n_obs_terms = n_sig_terms + 2 + self.n_actions # 2 for wealth and dt
            low = -np.inf * np.ones((n_obs_terms,), dtype=np.float32)
            low[-self.n_actions-2:-self.n_actions] = 0. # wealth and dt
        else:
            n_obs_terms = self.n_actions * self.window_len + 2 + self.n_actions # 2 for wealth and dt
            low = np.zeros((n_obs_terms,), dtype=np.float32)
        low[-self.n_actions:] = max_short
        high = np.inf * np.ones((n_obs_terms,), dtype=np.float32)
        high[-self.n_actions:] = max_long
        self.observation_space = spaces.Box(low=low, high=high, shape=(n_obs_terms,), dtype=np.float32)

    def reset(self):
        # observation must be a numpy array
        self.pm_env_reset(self.window_len - 1) # initial price of all 1s can form the first set of values in window_len
        return self.generate_obs()

    def step(self, action):
        reward, done, info = self.pm_env_step(action)
        return self.generate_obs(), reward, done, info

    def render(self):
        pass

    def close(self):
        pass

    def generate_obs(self):
        # Feed in prices of all assets based on window length and position'
        if self.real_data is None:
            window = np.array(self.episode_path[self.batch_period-(self.window_len-1):self.batch_period+1]).reshape(-1).astype(np.float32)
        else:
            window = np.array(self.episode_path[self.real_data_current_period-(self.window_len-1):self.real_data_current_period+1]).reshape(-1).astype(np.float32)
        # window has shape(window_len * n_actions,)

        if self.signature:
            start = self.batch_period-(self.window_len-1) if self.real_data is None else self.real_data_current_period-(self.window_len-1)
            end = self.batch_period+1 if self.real_data is None else self.real_data_current_period+1
            time = self.time[start:end]
            window = signature_transform(window, time, self.n_levels, self.lead_lag)

        obs = np.concatenate((window, [self.dt], [self.agent_wealth], self.position), axis=-1)
        return obs
