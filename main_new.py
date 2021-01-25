from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # using specific GPU
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

from compatible.likelihoods import MultiClass, Gaussian
from compatible.kernels import RBF, White
from gpflow.models.svgp import SVGP
from gpflow.training import AdamOptimizer, ScipyOptimizer
from scipy.stats import mode
from scipy.cluster.vq import kmeans2
import gpflow
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow import autoflow, params_as_tensors, ParamList
import pandas as pd
import itertools

pd.options.display.max_rows = 999
import gpflow_monitor

from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.io import loadmat
from gpflow_monitor import *

print('tf_ver:', tf.__version__, 'gpflow_ver:', gpflow.__version__)
from tensorflow.python.client import device_lib

print('avail devices:\n' + '\n'.join([x.name for x in device_lib.list_local_devices()]))
from jack_utils.common import time_it
import sys, pickle
import gpflow.training.monitor as mon
from numpy import exp, log

# our impl
from dgp_graph import *

import argparse

parser = argparse.ArgumentParser(description='main.')
parser.add_argument('--phase', type=int, default=0)
parser.add_argument('--maxiter', type=int, default=600)
parser.add_argument('--subn', type=int, default=100)
parser.add_argument('--matid', type=int, default=4)
parser.add_argument('--lb', type=int, default=5)
parser.add_argument('--lam', type=float, default=0.2)

args = parser.parse_args()

np.random.seed(123456)

maxiter = args.maxiter
lb = args.lb
model_vec = [lb]
_subn = args.subn

data_path1 = './stock_data/stock_phase%02d_lb%d.npz' % (args.phase, args.lb)
data1 = np.load(data_path1)

with open('./stock_data/data/relation/adj_mat.pkl', 'rb') as f:
    all_mat = pickle.load(f)


def get_sparsity(adj):
    avg_deg = np.mean([np.count_nonzero(x) for x in adj])
    return 100 * (1 - (np.count_nonzero(adj) / (adj.shape[0] ** 2))), avg_deg


adj0 = all_mat[args.matid]
trX0, trY0, teX0, teY0, choice = data1['rt_trainx'], data1['rt_trainy'], data1['rt_testx'], data1['rt_testy'], data1['choice']

ntrain = trX0.shape[0]
ntest = teX0.shape[0]
nodes = trY0.shape[1]


def normalize_data(data, mu, std):
    res = (data - mu) / std
    return res


def unnormalize_data(data, mu, std):
    res = data * std + mu
    return res


nodes = adj0.shape[0]


def dreshape(trX, trY, teX, teY, gmat, choice, subn=100):
    return trX.reshape([ntrain, nodes, -1])[:, choice[:subn], :], trY.reshape([ntrain, nodes, -1])[:, choice[:subn], :],\
           teX.reshape([ntest, nodes, -1])[:, choice[:subn], :], teY.reshape([ntest, nodes, -1])[:, choice[:subn], :], \
           gmat[choice[:subn], :][:, choice[:subn]]


subn = _subn
trX0, trY0, teX0, teY0, adj0 = dreshape(trX0, trY0, teX0, teY0, adj0, choice=choice, subn=subn)
nodes = subn

adj0 = adj0.astype(int)
print(adj0.shape)
adj0 += np.eye(nodes).astype(int)
adj0 = np.minimum(adj0, 1)

adj = adj0

mu_trX0, std_trX0 = np.mean(trX0, axis=0, keepdims=True), np.std(trX0, axis=0, keepdims=True)
mu_trY0, std_trY0 = np.mean(trY0, axis=0, keepdims=True), np.std(trY0, axis=0, keepdims=True)

trX = normalize_data(trX0, mu_trX0, std_trX0)
trY = normalize_data(trY0, mu_trY0, std_trY0)

perm = np.random.permutation(trX.shape[0])
trX = trX[perm]
trY = trY[perm]

teX = normalize_data(teX0, mu_trX0, std_trX0)
teY = normalize_data(teY0, mu_trY0, std_trY0)

M = 20

Z = np.stack([kmeans2(trX[:, i], M, minit='points')[0] for i in range(nodes)], axis=1)  # (M=s2=10, n, d_in=5)
print('inducing points Z: {}'.format(Z.shape))

adj = adj.astype('float64')
input_adj = adj  # adj  / np.identity(adj.shape[0]) /  np.ones_like(adj)

with gpflow.defer_build():
    m_dgpg = DGPG(trX, trY, Z, model_vec, Gaussian(), input_adj,
                  agg_op_name='concat3d', ARD=True,
                  is_Z_forward=True, mean_trainable=True, out_mf0=True,
                  num_samples=20, minibatch_size=40,
                  kern_type='RBF'
                  )
    # m_sgp = SVGP(X, Y, kernels, Gaussian(), Z=Z, minibatch_size=minibatch_size, whiten=False)
m_dgpg.compile()
model = m_dgpg

session = m_dgpg.enquire_session()
optimiser = gpflow.train.AdamOptimizer(0.01)
global_step = mon.create_global_step(session)

exp_path = "./exp/test"
# exp_path="./exp/temp"

print_task = mon.PrintTimingsTask() \
    .with_name('print') \
    .with_condition(mon.PeriodicIterationCondition(10)) \

checkpoint_task = mon.CheckpointTask(checkpoint_dir=exp_path)\
        .with_name('checkpoint') \
        .with_condition(mon.PeriodicIterationCondition(15))\

with mon.LogdirWriter(exp_path) as writer:
    tensorboard_task = mon.ModelToTensorBoardTask(writer, model) \
        .with_name('tensorboard') \
        .with_condition(mon.PeriodicIterationCondition(100)) \
        .with_exit_condition(True)
    monitor_tasks = []  # [print_task, tensorboard_task]

    with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
        optimiser.minimize(model, step_callback=monitor, global_step=global_step, maxiter=maxiter)

from jack_utils.my_metrics import *
import matplotlib.pyplot as plt


def assess_model_rmse(model, X_batch, Y_batch, S=10):
    m, v = model.predict_y(X_batch, S)
    pred = np.mean(m, axis=0)
    var = np.mean(v, axis=0)
    loss = np.sum((Y_batch.flatten() - pred.flatten()) ** 2)
    return loss, pred, var


def batch_assess_rmse(model, X, Y, batch_size=1, S=10):
    n_batches = max(int(len(X) / batch_size), 1)
    rms = len(X) - n_batches * batch_size
    losses, preds, varis = [], [], []
    Xr, Yr = X[-rms:, :], Y[-rms:, :]
    for X_batch, Y_batch in zip(np.split(X[:n_batches * batch_size], n_batches),
                                np.split(Y[:n_batches * batch_size], n_batches)):
        l, pred, vari = assess_model_rmse(model, X_batch, Y_batch, S=S)
        losses.append(l)
        preds.append(pred)
        varis.append(vari)
    if rms > 0:
        l, pred, vari = assess_model_rmse(model, Xr, Yr, S=S)
        losses.append(l)
        preds.append(pred)
        varis.append(vari)
    ndata = Y.shape[0] * Y.shape[1]
    avg_loss = np.sqrt(np.sum(losses) / ndata)
    y_pred = np.concatenate(preds)
    y_var = np.concatenate(varis)
    return avg_loss, y_pred, y_var


sample_X0, sample_Y0 = teX0, teY0.squeeze()
sample_X, sample_Y = teX, teY.squeeze()

model.trainable = False

for l in range(len(model_vec)):
    model.layers[l].kern.trainable = True

model.likelihood.trainable = True

optimiser = gpflow.train.AdamOptimizer(0.01)

exp_path += "-retrain_kern"
# exp_path="./exp/temp"

print_task = mon.PrintTimingsTask() \
    .with_name('print') \
    .with_condition(mon.PeriodicIterationCondition(10)) \

checkpoint_task = mon.CheckpointTask(checkpoint_dir=exp_path) \
    .with_name('checkpoint') \
    .with_condition(mon.PeriodicIterationCondition(15)) \

with mon.LogdirWriter(exp_path) as writer:
    tensorboard_task = mon.ModelToTensorBoardTask(writer, model) \
        .with_name('tensorboard') \
        .with_condition(mon.PeriodicIterationCondition(100)) \
        .with_exit_condition(True)
    monitor_tasks = [tensorboard_task, checkpoint_task]  # [print_task, tensorboard_task]

    with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
        optimiser.minimize(model, step_callback=monitor, global_step=global_step, maxiter=maxiter)

pred_rmse, predY, varY = batch_assess_rmse(model, sample_X.reshape(sample_X.shape[0], -1),
                                           sample_Y.reshape(sample_Y.shape[0], -1))

predY0 = unnormalize_data(predY[:, :, None], mu_trY0, std_trY0).squeeze()

metrics = [np_mae, np_rmse, np_mape]
e_dgp = [np.round(f(predY0, sample_Y0.squeeze()), 6) for f in metrics]
e_last = [np.round(f(sample_X0[:, :, -1], sample_Y0.squeeze()), 6) for f in metrics]
e_ha = [np.round(f(sample_X0.mean(axis=-1), sample_Y0.squeeze()), 6) for f in metrics]
e_mid = [np.round(f(np.median(sample_X0, axis=-1), sample_Y0.squeeze()), 6) for f in metrics]
print('metrics:\t[mae | rmse | mape]')
print('ours DGP:\t', e_dgp)
print('yesterday:\t', e_last)
print('day-mean:\t', e_ha)
print('day_median:\t', e_mid)

std_varY0 = np.sqrt(varY) * std_trY0.reshape(1, nodes)


def correct_rate(predY0, std_varY0, sample_Y0, ndev):
    predY0_ub = predY0 + std_varY0 * ndev
    predY0_lb = predY0 - std_varY0 * ndev
    tf_mat = np.logical_and(predY0_lb <= sample_Y0, sample_Y0 <= predY0_ub)
    correct_rate = np.sum(tf_mat) / np.product(tf_mat.shape)
    return correct_rate


print(correct_rate(predY0, std_varY0, sample_Y0, ndev=1))
print(correct_rate(predY0, std_varY0, sample_Y0, ndev=2))
print(correct_rate(predY0, std_varY0, sample_Y0, ndev=3))

# classification

node_avg_rt = trY0.reshape(-1)  # nodes

label_proportion = [1, 1, 1]

th_tot = np.sum(label_proportion)
tmp_rt = np.sort(node_avg_rt, axis=0)

tmp_th = 0
threshold = []
for th in label_proportion:
    threshold.append(tmp_rt[int(len(node_avg_rt) * float(th + tmp_th) / th_tot - 1)])
    tmp_th += th

bin_gt_neg = (sample_Y0 <= threshold[0])
bin_gt_neu = np.logical_and(threshold[0] < sample_Y0, sample_Y0 <= threshold[1])
bin_gt_pos = (threshold[1] < sample_Y0)

bin_pred_neg = (predY0 <= threshold[0])
bin_pred_neu = np.logical_and(threshold[0] < predY0, predY0 <= threshold[1])
bin_pred_pos = (threshold[1] < predY0)

# confusion matrix
from sklearn.metrics import confusion_matrix

y_true, y_pred = np.zeros(predY0.shape), np.zeros(predY0.shape)

y_true[bin_gt_neg] = 0
y_true[bin_gt_neu] = 1
y_true[bin_gt_pos] = 2

y_pred[bin_pred_neg] = 0
y_pred[bin_pred_neu] = 1
y_pred[bin_pred_pos] = 2

cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))

from sklearn.metrics import classification_report

print(classification_report(y_true.reshape(-1), y_pred.reshape(-1), digits=4, target_names=['pos', 'neu', 'neg']))

# return

## naive strategy

rt_v = []
ninvest = 10

for i in range(len(predY0)):
    p = predY0[i]
    sell_idx = np.argsort(p)[:ninvest // 2]
    buy_idx = np.argsort(-p)[:ninvest // 2]
    rr = 1 / ninvest
    tmp = rr * (sum(exp(sample_Y0[i, buy_idx])) + sum(exp(-sample_Y0[i, sell_idx])))
    rt_v.append(log(tmp))

print('naive strategy')
print('Total log return: %.8f' % sum(rt_v))
print('Total return: %.8f' % exp(sum(rt_v)))
print('Mean log return: %.8f' % np.mean(rt_v))
print('Mean return: %.8f' % exp(np.mean(rt_v)))

naive_rt = rt_v.copy()


## UCB strategy

def ucb_strategy(lam):
    rt_v = []
    ninvest = 10

    for i in range(len(predY0)):
        p = predY0[i]
        sell_idx = np.argsort(p - lam * std_varY0[i])[:ninvest // 2]
        buy_idx = np.argsort(-(p + lam * std_varY0[i]))[:ninvest // 2]
        rr = 1 / ninvest
        tmp = rr * (sum(exp(sample_Y0[i, buy_idx])) + sum(exp(-sample_Y0[i, sell_idx])))
        rt_v.append(log(tmp))
    return rt_v


rt_v = ucb_strategy(args.lam)

print('UCB strategy')
print('Total log return: %.8f' % sum(rt_v))
print('Total return: %.8f' % exp(sum(rt_v)))
print('Mean log return: %.8f' % np.mean(rt_v))
print('Mean return: %.8f' % exp(np.mean(rt_v)))

ucb_rt = rt_v.copy()

np.savez('./tmp/res_phase%02d_iter%d_subn%d_lb%d' % (args.phase, args.maxiter, args.subn, args.lb), predY0=predY0,
         std_varY0=std_varY0, sample_Y0=sample_Y0, naive_rt=naive_rt, ucb_rt=ucb_rt)

best_logrt = -np.inf
best_lam = 0

for lam in [1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.05, 0.03, 0.01, 0.001, 0]:
    rt_v = ucb_strategy(lam)
    print('lambda: %.3f   Total log return: %.8f' % (lam, sum(rt_v)))
    #     print('Total return: %.8f' % exp(sum(rt_v)))

    #     print('Mean log return: %.8f' % np.mean(rt_v))
    #     print('Mean return: %.8f' % exp(np.mean(rt_v)))

    if sum(rt_v) > best_logrt:
        best_logrt = sum(rt_v)
        best_lam = lam


