# -*- coding: utf-8 -*-
import os
import os.path as osp
import json
import random
import math
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import collections  # これを追加

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.autograd import Function, Variable
import torch.nn.functional as F
import torchvision.transforms

from torch.utils.data.sampler import Sampler, SubsetRandomSampler

from adapt.models.models import get_model
from adapt.solvers.solver import get_solver
import yaml
from tensorboardX import SummaryWriter
import matplotlib as mpl
from sklearn.manifold import TSNE
from pdb import set_trace
import copy
from time import time
import shutil

import pynvml
pynvml.nvmlInit()

import io
import os
import torch
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload
import seed
random.seed(seed.a)
np.random.seed(seed.a)
torch.manual_seed(seed.a)
torch.cuda.manual_seed_all(seed.a)  # CUDA全デバイスに対してシード設定
torch.cuda.manual_seed(seed.a)

from config.secrets import GOOGLE_CLIENT_SECRET_FILE, GOOGLE_TOKEN_FILE

SCOPES = ['https://www.googleapis.com/auth/drive']
CLIENT_SECRET_FILE = GOOGLE_CLIENT_SECRET_FILE
TOKEN_FILE = GOOGLE_TOKEN_FILE
DRIVE_ROOT_FOLDER_ID = '1mLWoJ41KzOh37csNhccw6uC1p5F8u7l3'  # Google DriveのルートフォルダID
######################################################################
##### Miscellaneous utilities and helper classes
######################################################################


class ActualSequentialSampler(Sampler):
	r"""Samples elements sequentially, always in the same order.

	Arguments:
		data_source (Dataset): dataset to sample from
	"""

	def __init__(self, data_source):
		self.data_source = data_source

	def __iter__(self):
		return iter(self.data_source)

	def __len__(self):
		return len(self.data_source)

######################################################################
##### Training utilities
######################################################################

class ReverseLayerF(Function):
	"""
	Gradient negation utility class
	"""				 
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg()
		return output, None

class ConditionalEntropyLoss(torch.nn.Module):
	"""
	Conditional entropy loss utility class
	"""				 
	def __init__(self):
		super(ConditionalEntropyLoss, self).__init__()

	def forward(self, x):
		b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
		b = b.sum(dim=1)
		return -1.0 * b.mean(dim=0)

######################################################################
##### Sampling utilities
######################################################################

def row_norms(X, squared=False):
	"""Row-wise (squared) Euclidean norm of X.
	Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
	matrices and does not create an X.shape-sized temporary.
	Performs no input validation.
	Parameters
	----------
	X : array_like
		The input array
	squared : bool, optional (default = False)
		If True, return squared norms.
	Returns
	-------
	array_like
		The row-wise (squared) Euclidean norm of X.
	"""
	norms = np.einsum('ij,ij->i', X, X)

	if not squared:
		np.sqrt(norms, norms)
	return norms

def get_embedding(model, loader, device, num_classes, args, with_emb=False, emb_dim=512):
	# model = model.to(device)
	model.eval()
	embedding = torch.zeros([len(loader.sampler), num_classes])
	embedding_pen = torch.zeros([len(loader.sampler), emb_dim])
	labels = torch.zeros(len(loader.sampler))
	preds = torch.zeros(len(loader.sampler))
	batch_sz = args.batch_size
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(loader):
			data, target = data.to(device), target.to(device)
			if with_emb:
				try:
					e1, e2 = model(data, with_emb=True)
				except StopIteration:
					print("data.shape model.device",data.shape)
				embedding_pen[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e2.shape[0]), :] = e2.cpu()
			else:
				e1 = model(data, with_emb=False)

			embedding[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0]), :] = e1.cpu()
			labels[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = target
			preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = e1.argmax(dim=1, keepdim=True).squeeze()

	return embedding, labels, preds, embedding_pen


def train(model, device, train_loader, optimizer, epoch):
	"""
	Test model on provided data for single epoch
	"""
	model.train()
	total_loss, correct = 0.0, 0
	for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
		if data.size(0) < 2: continue
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = nn.CrossEntropyLoss()(output, target)		
		total_loss += loss.item()
		pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
		corr =  pred.eq(target.view_as(pred)).sum().item()
		correct += corr
		loss.backward()
		optimizer.step()

	train_acc = 100. * correct / len(train_loader.sampler)
	avg_loss = total_loss / len(train_loader.sampler)
	print('\nTrain Epoch: {} | Avg. Loss: {:.3f} | Train Acc: {:.3f}'.format(epoch, avg_loss, train_acc))
	return avg_loss


def test(model, device, test_loader, mode="ori", split="test", topk=1):
	"""
	Test model on provided data
	"""
	print('\nEvaluating model on {}...'.format(split))
	model.eval()
	test_loss = 0
	correct = 0
	test_acc, topk_correct = 0, 0
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			if mode == "ori":
				output = model(data)
			elif mode == "brc":
				output = model(data)[0]
			if topk>1:
				_, topk_pred = torch.topk(output, topk, dim=1)  #只支持两个的元组
				topk_target = target.unsqueeze(1).repeat(1,int(topk))
				topk_corr = topk_pred.eq(topk_target).float().sum(dim=1).sum().item()
				topk_correct += topk_corr
			loss = nn.CrossEntropyLoss()(output, target) 
			test_loss += loss.item() # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			corr =  pred.eq(target.view_as(pred)).sum().item()
			correct += corr
			del loss, output
			# if batch_idx%100 == 0: print("iter batch idx:", batch_idx)
	test_loss /= len(test_loader.sampler)
	test_acc = 100. * correct / len(test_loader.sampler)
	topk_acc = 100. * topk_correct / len(test_loader.sampler) if topk > 1 else -1
	return test_acc, topk_acc

def test_per_class_accuracy(model, device, test_loader, mode="ori", split="test", class_index_to_name=None):
    """
    各クラスの正解率を計算する関数。

    Args:
        model: 評価するPyTorchモデル。
        device: モデルを実行するデバイス（例：'cpu' または 'cuda'）。
        test_loader: 評価対象のデータローダー。
        mode (str, optional): モード指定（デフォルトは "ori"）。
        split (str, optional): データの分割（デフォルトは "test"）。
        class_index_to_name (dict, optional): クラスインデックスからクラス名へのマッピング。

    Returns:
        dict: クラス名をキー、正解率（%）を値とする辞書。
    """
    model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_correct_array = np.zeros(50)
    class_total_array = np.zeros(50)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            if mode == "ori":
                output = model(data)
            elif mode == "brc":
                output = model(data)[0]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            _, predicted = torch.max(output, 1)

            for i in range(len(target)):
                label = target[i].item()
                pred = predicted[i].item()
                if pred == label:
                    class_correct[label] += 1
                    class_correct_array[int(label)] += 1
                class_total[label] += 1
                class_total_array[int(label)] += 1

    per_class_accuracy = {}
    for label in class_total:
        accuracy = 100.0 * class_correct[label] / class_total[label] if class_total[label] > 0 else 0.0
        if class_index_to_name is not None:
            class_name = class_index_to_name.get(str(label), f"Class {label}")
        else:
            class_name = f"Class {label}"
        per_class_accuracy[class_name] = accuracy
        
    per_class_accuracy_array = np.zeros(50)
    class_total_array_safe = np.where(class_total_array == 0, 1, class_total_array)
    per_class_accuracy_array = (class_correct_array / class_total_array_safe) * 100  

    return per_class_accuracy, per_class_accuracy_array


def save_instance_outputs_with_paths(model, device, data_loader, output_file='test_results.csv', topk=1):
    """
    モデルを評価し、各インスタンスの情報を保存するとともに、全体の精度を計算・表示する関数。

    Args:
        model: 評価するPyTorchモデル。
        device: モデルを実行するデバイス（例：'cpu' または 'cuda'）。
        data_loader: 評価対象のデータローダー（画像パスを含む）。
        output_file: 保存するCSVファイルの名前。
        topk: トップK精度を計算する場合のKの値。

    Returns:
        None
    """
    model.eval()
    results = []
    correct = 0
    total = 0
    topk_correct = 0

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                data, target, paths = batch
            elif len(batch) == 4:
                # __getitem_with_path__ が 4要素を返す場合（例：強いデータ拡張あり）
                data, _, target, paths = batch
            else:
                raise ValueError("Unexpected number of elements in data tuple.")

            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            probabilities = nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # 全体の精度を計算
            correct += (predicted == target).sum().item()
            total += target.size(0)

            # トップK精度を計算
            if topk > 1:
                _, topk_pred = torch.topk(outputs, topk, dim=1)
                topk_correct += topk_pred.eq(target.view(-1, 1).expand_as(topk_pred)).sum().item()

            # 各インスタンスの情報を保存
            for i in range(data.size(0)):
                result = {
                    'image_path': paths[i],  # 画像パス
                    'true_label': target[i].item(),  # 真のラベル
                    'predicted_label': predicted[i].item(),  # 予測ラベル
                    'probabilities': probabilities[i].cpu().numpy().tolist()  # 各クラスに対する確率
                }
                results.append(result)

    # 結果をDataFrameに変換してCSVに保存
    df = pd.DataFrame(results)
    # 'probabilities' 列をカンマ区切りの文字列に変換
    df['probabilities'] = df['probabilities'].apply(lambda x: ','.join(map(str, x)))
    df.to_csv(output_file, index=False)
    print(f'Results saved to {output_file}')

    # 精度を計算して表示
    accuracy = 100. * correct / total if total > 0 else 0
    topk_accuracy = 100. * topk_correct / total if topk > 1 and total > 0 else -1

    print(f'Accuracy: {accuracy:.2f}%')
    if topk > 1:
        print(f'Top-{topk} Accuracy: {topk_accuracy:.2f}%')

######################################################################
##### Interactive visualization utilities
######################################################################

def log(target_accs, fname):
	"""
	Log results as JSON
	"""
	with open(fname, 'w') as f:
		json.dump(target_accs, f, indent=4)


def gen_dir(args):
	exp_name = '{}_{}_{}_{}r_{}b_{}'.format(args.model_init, args.al_strat, args.da_strat, \
											args.num_rounds, int(args.total_budget), args.cnn)
	
	arg_str =  'warmup{}-adapt_lr{}-wd{}'.format(args.warm_strat, args.adapt_lr, args.wd)		
	if args.da_strat == "self_ft":
		arg_str += '-srcw{:.1f}-ccw{:.1f}-ucw{:.1f}'.format(args.src_weight, args.cc_weight, args.uc_weight) 

	#exps_path = osp.join('exp_same_round_diff_budget', args.round_type, args.dataset, args.id, exp_name, arg_str) 
 	#exps_path = osp.join('exp_record', args.round_type, args.dataset, args.id, exp_name, arg_str) 
	#exps_path = osp.join('only_tgt_sup_exp_record_6r_150b_fix_seed_not_augment', args.round_type, args.dataset, args.id, exp_name, arg_str, f"subset_{args.subset_idx}",args.sampling) 
	exps_path = osp.join('with_gmm_only_tgt_sup_exp_record_6r_150b_fix_seed_not_augment', args.round_type, args.dataset, args.id, exp_name, arg_str, f"subset_{args.subset_idx}",args.sampling) 	
	os.makedirs(exps_path, exist_ok=True) 
	run_num = 1 if not os.listdir(exps_path) else np.array([int(i) for i in os.listdir(exps_path) if '.txt' not in i]).max()+1

	exp_path = osp.join(exps_path, str(run_num) )
	if osp.exists(exp_path):
		set_trace();print("press c to dele exited run path: ", exp_path);shutil.rmtree(exp_path)
	os.makedirs(exp_path, exist_ok=True)
	
	with open(osp.join(exp_path, 'config.yaml'), 'w') as f:
		yaml.dump(dict(args),f)
	writer = SummaryWriter(exp_path)
	
	return writer, exp_path, run_num


def topk_feat_pred(logits, embs, cls_pro, k_feat=32, k_pred=10):
	ulb_num, emb_dim = embs.shape[0], embs.shape[1]
	_, embs_max_idx = torch.topk(embs, k_feat, dim=1)  #N,D  N,k_feat
	sort_embs_max_idx, _ = torch.sort(embs_max_idx, dim=1)
	
	_, pros_max_idx = torch.topk(cls_pro, k_feat, dim=1)  #C,D  C,k_feat
	sort_pros_max_idx, _ = torch.sort(pros_max_idx, dim=1)

	fixed_pred = torch.zeros(ulb_num)
	s_time = time()
	for i in range(ulb_num):
		# if i % 5000 == 0: print("-----now i is ",i)
		emb_i = sort_embs_max_idx[i]
		_, topk_pred_idxs = torch.topk(logits[i], k_pred)  
		candi_pros = sort_pros_max_idx[topk_pred_idxs] 
		candi_sims = torch.zeros(k_pred)
		for j in range(k_pred):
			candi_sims[j] = cal_iou(emb_i, candi_pros[j])  
		idx_in_candi = candi_sims.argmax()
		fixed_pred[i] = topk_pred_idxs[idx_in_candi]
	# print("topk_feat_pred takes mins:", (time()-s_time)//60)
	return fixed_pred

def cal_iou(a,b):
	a, b = set(a.cpu().numpy()), set(b.cpu().numpy())
	return len(a&b)/len(a|b)
    
def get_conf_balance_for_subset_majority_at_least_one(tgtuns_logits, gmm_confs, min_num, class_num, subset_classes):
    """
    指定されたクラス部分集合に対してバランスを取り、余った数を他のクラスから選択する。

    Args:
        tgtuns_logits (torch.Tensor): 未ラベルデータのロジット [N, class_num]
        gmm_confs (np.array): GMM による信頼度 [N]
        min_num (int): サンプルの総数
        subset_classes (np.array): 部分集合クラスのインデックス (1次元)

    Returns:
        np.array: 選択されたサンプルのインデックス
    """
    # tgtuns_logits から予測ラベルを取得
    tgtuns_preds = torch.argmax(tgtuns_logits, dim=1)  # [N]

    # 全クラス数
    class_num = tgtuns_logits.size(1)

    # 部分集合以外のクラスを特定
    non_subset_classes = np.setdiff1d(np.arange(class_num), subset_classes)

    # 部分集合クラスの数
    subset_class_num = len(subset_classes)

    candi_idx = []

    # ステップ 1: 部分集合クラスから1つずつ選択
    for cls in subset_classes:
        pred_cls_idx = torch.where(tgtuns_preds == cls)[0].cpu().numpy()
        if len(pred_cls_idx) == 0:
            continue
        # 信頼度が高い順に1個選択
        sorted_idx = gmm_confs[pred_cls_idx].argsort()[::-1]
        selected_idx = pred_cls_idx[sorted_idx[:1]]
        candi_idx.append(selected_idx)

    # ステップ 2: 部分集合以外のクラスから1つずつ選択
    for cls in non_subset_classes:
        if len(candi_idx) >= min_num:
            break
        pred_cls_idx = torch.where(tgtuns_preds == cls)[0].cpu().numpy()
        if len(pred_cls_idx) == 0:
            continue
        # 信頼度が高い順に1個選択
        sorted_idx = gmm_confs[pred_cls_idx].argsort()[::-1]
        selected_idx = pred_cls_idx[sorted_idx[:1]]
        candi_idx.append(selected_idx)

    # ステップ 3: 残りの数を部分集合クラスからバランスを取って選択
    remaining_num = min_num - len(candi_idx)
    if remaining_num > 0:
        num_per_subset_class = remaining_num // subset_class_num
        leftover_subset = remaining_num % subset_class_num

        for cls in subset_classes:
            if len(candi_idx) >= min_num:
                break
            pred_cls_idx = torch.where(tgtuns_preds == cls)[0].cpu().numpy()
            if len(pred_cls_idx) == 0:
                continue

            # 信頼度が高い順に num_per_subset_class 個選択
            sorted_idx = gmm_confs[pred_cls_idx].argsort()[::-1]
            selected_idx = pred_cls_idx[sorted_idx[:num_per_subset_class]]
            candi_idx.append(selected_idx)

        # leftover を部分集合クラスに分配
        if leftover_subset > 0:
            for cls in subset_classes:
                if len(candi_idx) >= min_num:
                    break
                pred_cls_idx = torch.where(tgtuns_preds == cls)[0].cpu().numpy()
                if len(pred_cls_idx) == 0:
                    continue

                remaining_idx = np.setdiff1d(pred_cls_idx, np.concatenate(candi_idx, axis=0), assume_unique=True)

                if len(remaining_idx) == 0:
                    continue

                sorted_idx = gmm_confs[remaining_idx].argsort()[::-1]
                selected_idx = remaining_idx[sorted_idx[:leftover_subset]]
                candi_idx.append(selected_idx)
                leftover_subset -= len(selected_idx)

                if leftover_subset <= 0:
                    break

    # インデックスを統合して返す
    if len(candi_idx) > 0:
        return np.concatenate(candi_idx, axis=0)[:min_num]  # 必ず min_num に調整
    else:
        return np.array([], dtype=int)



def get_disc(input_dim):
	disc =  nn.Sequential(
					nn.Linear(input_dim, 500),
					nn.ReLU(),
					nn.Linear(500, 500),
					nn.ReLU(),
					nn.Linear(500, 2),
					)
	return disc

def get_gpu_usedrate(need_gpu_count=1):
    used_rates = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used= float(meminfo.used/meminfo.total    )
        used_rates.append(used)
    used_num = np.array(used_rates).argsort()[:need_gpu_count]
    used_str = ','.join(map(str, used_num))
    return used_str, used_num

# for BADGE AL
def outer_product_opt(c1, d1, c2, d2):
	"""Computes euclidean distance between a1xb1 and a2xb2 without evaluating / storing cross products
	"""
	B1, B2 = c1.shape[0], c2.shape[0]
	t1 = np.matmul(np.matmul(c1[:, None, :], c1[:, None, :].swapaxes(2, 1)), np.matmul(d1[:, None, :], d1[:, None, :].swapaxes(2, 1)))
	t2 = np.matmul(np.matmul(c2[:, None, :], c2[:, None, :].swapaxes(2, 1)), np.matmul(d2[:, None, :], d2[:, None, :].swapaxes(2, 1)))
	t3 = np.matmul(c1, c2.T) * np.matmul(d1, d2.T)
	t1 = t1.reshape(B1, 1).repeat(B2, axis=1)
	t2 = t2.reshape(1, B2).repeat(B1, axis=0)
	return t1 + t2 - 2*t3

def kmeans_plus_plus_opt(X1, X2, n_clusters, init=[0], random_state=np.random.RandomState(seed.a), n_local_trials=None):
	"""Init n_clusters seeds according to k-means++ (adapted from scikit-learn source code)
	Parameters
	----------
	X1, X2 : array or sparse matrix
		The data to pick seeds for. To avoid memory copy, the input data
		should be double precision (dtype=np.float64).
	n_clusters : integer
		The number of seeds to choose
	init : list
		List of points already picked
	random_state : int, RandomState instance
		The generator used to initialize the centers. Use an int to make the
		randomness deterministic.
		See :term:`Glossary <random_state>`.
	n_local_trials : integer, optional
		The number of seeding trials for each center (except the first),
		of which the one reducing inertia the most is greedily chosen.
		Set to None to make the number of trials depend logarithmically
		on the number of seeds (2+log(k)); this is the default.
	Notes
	-----
	Selects initial cluster centers for k-mean clustering in a smart way
	to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
	"k-means++: the advantages of careful seeding". ACM-SIAM symposium
	on Discrete algorithms. 2007
	Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
	which is the implementation used in the aforementioned paper.
	"""

	n_samples, n_feat1 = X1.shape
	_, n_feat2 = X2.shape
	# x_squared_norms = row_norms(X, squared=True)
	centers1 = np.empty((n_clusters+len(init)-1, n_feat1), dtype=X1.dtype)
	centers2 = np.empty((n_clusters+len(init)-1, n_feat2), dtype=X1.dtype)

	idxs = np.empty((n_clusters+len(init)-1,), dtype=np.long)

	# Set the number of local seeding trials if none is given
	if n_local_trials is None:
		# This is what Arthur/Vassilvitskii tried, but did not report
		# specific results for other than mentioning in the conclusion
		# that it helped.
		n_local_trials = 2 + int(np.log(n_clusters))

	# Pick first center randomly
	center_id = init

	centers1[:len(init)] = X1[center_id]
	centers2[:len(init)] = X2[center_id]
	idxs[:len(init)] = center_id

	# Initialize list of closest distances and calculate current potential
	distance_to_candidates = outer_product_opt(centers1[:len(init)], centers2[:len(init)], X1, X2).reshape(len(init), -1)

	candidates_pot = distance_to_candidates.sum(axis=1)
	best_candidate = np.argmin(candidates_pot)
	current_pot = candidates_pot[best_candidate]
	closest_dist_sq = distance_to_candidates[best_candidate]

	# Pick the remaining n_clusters-1 points
	for c in range(len(init), len(init)+n_clusters-1):
		# Choose center candidates by sampling with probability proportional
		# to the squared distance to the closest existing center
		rand_vals = random_state.random_sample(n_local_trials) * current_pot
		candidate_ids = np.searchsorted(closest_dist_sq.cumsum(),
										rand_vals)
		# XXX: numerical imprecision can result in a candidate_id out of range
		np.clip(candidate_ids, None, closest_dist_sq.size - 1,
				out=candidate_ids)

		# Compute distances to center candidates
		distance_to_candidates = outer_product_opt(X1[candidate_ids], X2[candidate_ids], X1, X2).reshape(len(candidate_ids), -1)

		# update closest distances squared and potential for each candidate
		np.minimum(closest_dist_sq, distance_to_candidates,
				   out=distance_to_candidates)
		candidates_pot = distance_to_candidates.sum(axis=1)

		# Decide which candidate is the best
		best_candidate = np.argmin(candidates_pot)
		current_pot = candidates_pot[best_candidate]
		closest_dist_sq = distance_to_candidates[best_candidate]
		best_candidate = candidate_ids[best_candidate]

		idxs[c] = best_candidate

	return None, idxs[len(init)-1:]


def read_model_drive(round_num, args):
    # Google Drive APIの認証とサービス構築
    creds = authenticate_google_drive()
    service = build('drive', 'v3', credentials=creds)

    # モデルを初期化
    da_model = get_model('ResNet34', num_cls=args.num_classes).to(device)

    # Google Drive上のモデルファイルのパスを構築
    drive_model_dir = os.path.join(
        f'with_gmm_only_tgt_sup_models_fix_seed_not_augment_{args.cnn}', args.dataset,
        f"{args.source_domain}2{args.target_domain}", f"{args.subset_idx}s_{int(args.total_budget)}b_{args.num_rounds}r",
        f"{args.sampling}", f"{args.run_num}"
    )
    model_file_name = f"round_{round_num}.pth"
    print()
    # Google Drive内のディレクトリ構造を検索
    model_file_id = find_file_in_drive(service, model_file_name, drive_model_dir, DRIVE_ROOT_FOLDER_ID)
    if not model_file_id:
        print(f"Model file not found in Google Drive for round {round_num}: {model_file_name}")
        return None

    # モデルファイルを一時的にローカルにダウンロード
    local_model_path = f"/tmp/{model_file_name}"
    download_file_from_drive(service, model_file_id, local_model_path)

    # モデルの状態をロード
    state_dict = torch.load(local_model_path, map_location=device)
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    # strict=Trueでパラメータが完全一致するか確認
    try:
        da_model.load_state_dict(state_dict, strict=True)
        print(f"Model for round {round_num}, {args.source_domain}2{args.target_domain}, {args.subset_idx}, {args.sampling} loaded successfully with strict=True.")
    except RuntimeError as e:
        print(f"Error loading model for round {round_num} with strict=True: {e}")
        return None

    da_model.eval()
    return da_model

def authenticate_google_drive():
    creds = None

    # 保存されたトークンを読み込む
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # トークンが無効または存在しない場合、再認証
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
            auth_url, _ = flow.authorization_url(prompt='consent')
            print("以下のURLをブラウザで開いて認証してください:")
            print(auth_url)
            auth_code = input("認証コードを入力してください: ")
            flow.fetch_token(code=auth_code)
            creds = flow.credentials

        # 新しいトークンを保存
        with open(TOKEN_FILE, 'w') as token_file:
            token_file.write(creds.to_json())

    return creds

def find_folder_id_by_path(service, folder_path, root_folder_id):
    """
    Google Driveのフォルダパスを辿り、最終フォルダのIDを返す
    """
    current_folder_id = root_folder_id
    folders = folder_path.split('/')  # フォルダ階層を分割

    for folder_name in folders:
        query = f"'{current_folder_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        print(f"Searching for folder: {folder_name} with query: {query}")
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print(f"Folder not found: {folder_name}")
            return None  # フォルダが見つからない場合はNoneを返す
        
        current_folder_id = items[0]['id']  # 見つかったフォルダのIDを取得

    return current_folder_id

def find_file_in_drive(service, file_name, folder_path, root_folder_id):
    # フォルダ階層を検索して最終フォルダのIDを取得
    folder_id = find_folder_id_by_path(service, folder_path, root_folder_id)
    if not folder_id:
        print(f"Folder path not found: {folder_path}")
        return None

    # フォルダ内で対象ファイルを検索
    query = f"'{folder_id}' in parents and name='{file_name}'"
    print(f"Searching for file: {file_name} with query: {query}")
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print(f"File not found: {file_name} in folder {folder_path}")
        return None

    file_id = items[0]['id']
    print(f"Found file: {file_name} with ID: {file_id}")
    return file_id


def download_file_from_drive(service, file_id, destination):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}% complete.")

def get_all_labels_and_predictions(model, device, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def save_model_to_drive(service, model, model_dir, file_name, root_folder_id):
    """
    Google Drive上にフォルダ階層を作成し、PyTorchモデルを保存する。

    Parameters:
    - service: Google Drive APIのサービスオブジェクト
    - model: 保存するPyTorchモデル（torch.nn.Module）またはstate_dict（OrderedDict）
    - model_dir: Google Drive上で保存するフォルダパス（例: "parent_folder/child_folder"）
    - file_name: 保存するファイル名（例: "round_1.pth"）
    - root_folder_id: Google DriveのルートフォルダID

    Returns:
    - 保存されたファイルのID
    """
    def delete_existing_file(service, folder_id, file_name):
        # フォルダ内の同名ファイルを検索
        query = f"'{folder_id}' in parents and name = '{file_name}' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])

        for file in files:
            # 同名ファイルを削除
            service.files().delete(fileId=file['id']).execute()
            print(f"Deleted existing file: {file['name']} (ID: {file['id']})")

    # Google Drive上にフォルダ階層を作成
    folder_id = create_folders_in_drive(service, model_dir, root_folder_id)

    # 同名ファイルを削除
    delete_existing_file(service, folder_id, file_name)

    # モデルがstate_dict形式ではない場合はstate_dictを取得
    if isinstance(model, torch.nn.Module):
        state_dict = model.state_dict()
    elif isinstance(model, collections.OrderedDict):
        state_dict = model
    else:
        raise ValueError("The model must be a torch.nn.Module or a state_dict (OrderedDict)")

    # 一時的なローカルファイルにモデルを保存
    temp_model_path = f"/tmp/{file_name}"
    torch.save(state_dict, temp_model_path)

    # Google Driveにモデルファイルをアップロード
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(temp_model_path, resumable=True)
    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(f"Model saved to Google Drive: {model_dir}/{file_name} (File ID: {uploaded_file['id']})")

    # 一時ファイルを削除
    os.remove(temp_model_path)

    return uploaded_file['id']

def create_folders_in_drive(service, folder_path, root_folder_id):
    """
    指定されたフォルダパスに従い、必要なフォルダだけをGoogle Driveに作成します。

    Parameters:
    - service: Google Drive APIのサービスオブジェクト
    - folder_path: Google Drive内のフォルダ階層を表す文字列（例: "parent_folder/child_folder"）
    - root_folder_id: フォルダのルートID（Driveのベースフォルダ）

    Returns:
    - 最下層のフォルダID
    """
    current_folder_id = root_folder_id
    folders = folder_path.split('/')  # フォルダ階層を分割

    for folder_name in folders:
        # 現在のフォルダ内で対象フォルダを検索
        query = f"'{current_folder_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])

        if items:
            # フォルダが既に存在する場合
            current_folder_id = items[0]['id']
            #print(f"Folder exists: {folder_name} (ID: {current_folder_id})")
        else:
            # フォルダが存在しない場合、新規作成
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [current_folder_id]
            }
            folder = service.files().create(body=folder_metadata, fields='id').execute()
            current_folder_id = folder['id']
            #print(f"Created folder: {folder_name} (ID: {current_folder_id})")

    return current_folder_id

if __name__ == "__main__":
    used_str, used_num = get_gpu_usedrate(need_gpu_count=1)
    print(int(used_num[0]))