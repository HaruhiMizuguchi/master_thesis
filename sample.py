# -*- coding: utf-8 -*-
import os
import copy
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

from data import *
import utils as utils
from utils import ActualSequentialSampler
from adapt.solvers.solver import get_solver

from pdb import set_trace

from time import time
import os
from adapt.models.task_net import ss_GaussianMixture
import active_learning

from torch.cuda.amp import autocast, GradScaler
import seed
torch.manual_seed(seed.a)
torch.cuda.manual_seed(seed.a)
torch.cuda.manual_seed_all(seed.a)  # CUDA全デバイスに対してシード設定
random.seed(seed.a)
np.random.seed(seed.a)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

al_dict = {}
def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls
    return decorator

def get_strategy(sample, *args):
	if sample not in al_dict: raise NotImplementedError
	return al_dict[sample](*args)

class_num = {"officehome":65, "domainnet":345, "cifar10":10 , "domainnet_50":50}

class SamplingStrategy:
	""" 
	Sampling Strategy wrapper class
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, writer, run, exp_path):
		self.dset = dset
		self.num_classes = class_num[args.dataset]
		self.train_idx = np.array(train_idx)
		self.model = model
		if discriminator is not None:
			self.discriminator = discriminator.to(device) 
		else: 
			self.discriminator = discriminator
		self.device = device
		self.args = args
		self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)  

		# added on CLUE's code
		self.writer = writer
		self.run = run
		self.exp_path = exp_path
		self.gc_model = None
		self.query_count = 0
		self.loss_p = os.path.join(self.exp_path, 'loss')
		os.makedirs(self.loss_p, exist_ok=True)
		if self.args.cnn == 'LeNet':
			self.emb_dim = 500
		elif self.args.cnn in ['ResNet34']:
			self.emb_dim = 512
		elif self.args.cnn in ['ResNet50']:
			self.emb_dim = 2048
		elif self.args.cnn in ['ResNet50_FE']:
			self.emb_dim = 256
		else: raise NotImplementedError


	def query(self, n, src_loader):
		pass

	def update(self, idxs_lb):
		self.idxs_lb = idxs_lb
	
	def train(self, target_train_dset, args, src_loader=[], tgt_conf_loader=[], tgt_unconf_loader=[]):
		"""	
		Driver train method: using current all data to train in a semi-surpervised way
		"""

		train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])  
		
		tgt_sup_loader = torch.utils.data.DataLoader(target_train_dset, sampler=train_sampler, num_workers=args.num_workers, \
													batch_size=args.batch_size, drop_last=False)    # target lab
		def seed_worker(worker_id):
			worker_seed = seed.a + worker_id
			np.random.seed(worker_seed)
			random.seed(worker_seed)
		tgt_unsup_loader = torch.utils.data.DataLoader(target_train_dset, shuffle=True, num_workers=args.num_workers, \
													   batch_size=args.batch_size, drop_last=False, worker_init_fn=seed_worker) # target lab+unlab
 
		"""for data,labels in tgt_sup_loader:
			print("labels",labels)
			logits = self.model(data)
			print("logits",logits)
			print(logits.shape)
			preds = torch.argmax(logits, dim=1)
			print("preds",preds)
		"""	
		opt_net_tgt = optim.Adam(self.model.parameters(), lr=args.adapt_lr, weight_decay=args.wd)
		lr_scheduler = optim.lr_scheduler.StepLR(opt_net_tgt, 20, 0.5)
		scaler = GradScaler()

		solver = get_solver(args.da_strat, self.model, src_loader, tgt_sup_loader, tgt_unsup_loader, \
						self.train_idx, opt_net_tgt, self.query_count, self.device, self.args, self.run)

		round_iter, qc_best_acc = 0, -1  # Iteration of this round (args.adapt_num_epochs epochs)
		################################################################
		# ラベル付きデータのラベル分布を表示
		# ラベル付きデータのインデックスを取得
		labeled_idxs = self.train_idx[self.idxs_lb]
		
		# ラベルを集計するためにtarget_train_dsetからラベルを取得
		labeled_labels = [target_train_dset[i][1] for i in labeled_idxs]
		
		# ラベルの分布をカウント
		unique_labels, label_counts = np.unique(labeled_labels, return_counts=True)
		
		# ラベル分布を表示
		#print("ラベル付きデータのラベル分布:")
		#for label, count in zip(unique_labels, label_counts):
		#		print(f"ラベル {label}: {count}個")
		#################################################################
		for epoch in range(args.adapt_num_epochs):  
			if args.da_strat == 'ft':
				round_iter = solver.solve(epoch, self.writer, round_iter)
			elif args.da_strat == 'mme':
				round_iter = solver.solve(epoch, self.writer, round_iter)  
			elif args.da_strat == 'dann':
				opt_dis_adapt = optim.Adam(self.discriminator.parameters(), lr=args.adapt_lr, betas=(0.9, 0.999), weight_decay=0)
				solver.solve(epoch, self.discriminator, opt_dis_adapt)			
			elif args.da_strat == 'self_ft':
				if args.iter_num == "tgt_sup_loader":
					iter_num = len(tgt_sup_loader) 
				elif args.iter_num == "tgt_conf_loader":
					iter_num = len(tgt_conf_loader) 
				else: raise NotImplementedError
				iter_num = iter_num * args.iter_rate
				if args.loss == "all":
					round_iter = solver.solve_common_amp(epoch, self.writer, round_iter, tgt_conf_loader, tgt_unconf_loader, iter_num, scaler, args.loss,args.sampling,
																gmm1_train=True, conf_mask=False)
				elif args.loss == "only_sup":
					round_iter = solver.solve_common_amp(epoch, self.writer, round_iter, tgt_conf_loader, tgt_unconf_loader, iter_num, scaler, args.loss,args.sampling,
																gmm1_train=True, conf_mask=False)
			else: raise NotImplementedError
			lr_scheduler.step()

		return self.model, qc_best_acc
	
	def get_embed(self, src_loader):
		# 1.compute z* in tgt supervised and source dataset with shape[num_class,embedding_dim]
		emb_dim = self.emb_dim
		# source data emb
		src_logits, src_lab, src_preds, src_emb = utils.get_embedding(self.model, src_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)

		# target labeled data emb
		idxs_labeled = np.arange(len(self.train_idx))[self.idxs_lb]
		tgts_sampler = ActualSequentialSampler(self.train_idx[idxs_labeled])
		tgts_loader = torch.utils.data.DataLoader(self.dset, sampler=tgts_sampler, num_workers=self.args.num_workers, \
												  batch_size=self.args.batch_size, drop_last=False)
		tgts_logits, tgts_lab, tgts_preds, tgts_emb = utils.get_embedding(self.model, tgts_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)															   
		
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]  
		tgtuns_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		tgtuns_loader = torch.utils.data.DataLoader(self.dset, sampler=tgtuns_sampler, num_workers=self.args.num_workers, \
												  batch_size=self.args.batch_size, drop_last=False)
										  
		# target unlabeled data emb
		tgtuns_logits , tgtuns_lab , tgtuns_preds, tgtuns_emb = utils.get_embedding(self.model, tgtuns_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)
		return tgts_logits, tgts_lab, tgts_preds, tgts_emb, tgtuns_logits , tgtuns_lab , tgtuns_preds, tgtuns_emb, src_logits, src_lab, src_preds, src_emb, idxs_unlabeled


	# Calculate category-wise centroids
	def calpro_fixpred(self, src_lab,  src_pen_emb, tgtuns_logits , tgtuns_pen_emb, k_feat, tgts_lab=[], tgts_pen_emb=[]):
		emb_dim = self.emb_dim

		cls_prototypes = torch.zeros([self.num_classes, emb_dim])
		tgtuns_preds = torch.argmax(tgtuns_logits, dim=1)
		for i in range(self.num_classes):
			anchor_i = src_pen_emb[src_lab == i]
			if self.query_count > 1:
				emb = tgts_pen_emb[tgts_lab == i] 
				if len(emb) > 0: anchor_i = torch.cat([anchor_i, emb],dim=0)
			anchor_i = anchor_i.mean(dim=0).reshape(-1)
			cls_prototypes[i,:] = anchor_i
		
		fixed_unstgt_preds = utils.topk_feat_pred(tgtuns_logits, tgtuns_pen_emb, cls_prototypes, k_feat= k_feat, k_pred=self.num_classes)
		return fixed_unstgt_preds


@register_strategy('GMM')
class GMM(SamplingStrategy):
	def __init__(self, dset, train_idx, model, discriminator, device, args, writer, run, exp_path):
		super(GMM, self).__init__(dset, train_idx, model, discriminator, device, args, writer, run, exp_path)
		self.GMM_models = {}
		self.loss_type = "fix_psd"
		self.qc1_sele = True 
		self.qc_conf_type = "conf_thred" 
		self.post_conf = "max" 

	def query(self, n, src_loader):
		# クエリ回数をインクリメント
		self.query_count += 1
		# 現在のランとクエリ回数を表示
		print('-------Run:{}/query_count:{}/ start--------'.format(self.run+1, self.query_count))
		
		#------GMM 学習------
		#1.ターゲットの教師ありデータとソースデータセットにおいて、z* (クラス中心) を計算 [クラス数, 埋め込み次元]
		emb_dim = self.emb_dim
		## ソースデータ
		# ソースデータの埋め込み、ラベル、予測、ペナルティ項付き埋め込みを取得
		src_logits, src_lab, src_preds, src_pen_emb = utils.get_embedding(self.model, src_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)												   
		## ターゲットのラベルなしデータ
		# ラベルなしデータのインデックスを取得
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]  # ラベルなしターゲット訓練セット [0,1,..,len(U)-1]
		# ラベルなしデータ用のサンプラーを作成
		tgtuns_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		# ラベルなしデータ用のデータローダーを作成
		tgtuns_loader = torch.utils.data.DataLoader(self.dset, sampler=tgtuns_sampler, num_workers=4, batch_size = self.args.batch_size, drop_last=False)
		# ラベルなしターゲットデータの埋め込み、ラベル、予測、ペナルティ項付き埋め込みを取得
		tgtuns_logits , tgtuns_lab , tgtuns_preds, tgtuns_pen_emb = utils.get_embedding(self.model, tgtuns_loader, self.device, \
																	   self.num_classes, self.args, with_emb=True, emb_dim=emb_dim)
																	   
		# クロスエントロピー損失関数を定義 (要素ごとの損失を計算)
		every_loss =  nn.CrossEntropyLoss(reduction="none")

		######################################################################
		##### 変数命名規則
		##### -- cc(confident-consistent:確信あり-一致), uc(uncertain-consistent:確信なし-一致), ui(uncertain-inconsistent:確信なし-不一致), ci(confident-inconsistent:確信あり-不一致)
		##### -- S(ソース), T(ターゲットラベルあり), U(ターゲットラベルなし)
		##### -- ''*index'' はUST集合におけるインデックスを意味する
		######################################################################
		# STci_loss, STcc_loss, STui_loss, STuc_loss, U_loss, loss_assist_ALL = [], [], [], [], [], []
		# 各損失のリストを初期化
		STci_loss, STcc_loss, STui_loss, STuc_loss, U_loss = [], [], [], [], []
		loss_lab, s_time = [], time()

		# ソースデータのラベル
		ST_y = src_lab
		# ソースとターゲットラベルなしデータのロジットを結合
		UST_logits = torch.cat([tgtuns_logits, src_logits], dim=0)   
		# ソースとターゲットラベルなしデータの予測ラベルを結合
		UST_plab = torch.cat([tgtuns_preds, src_preds])  
		# ソースとターゲットラベルなしデータの真のラベル（ソースは真のラベル、ターゲットラベルなしは予測ラベル）を結合
		UST_label = torch.cat([tgtuns_preds, src_lab], dim=0)  
		tgts_lab, tgts_pen_emb = [], []

		# 2回目以降のクエリの場合、ラベルありターゲットデータも考慮
		if self.query_count > 1:
			# ラベルありデータのインデックスを取得
			idxs_labeled = np.arange(len(self.train_idx))[self.idxs_lb]
			# ラベルありデータ用のサンプラーを作成
			tgts_sampler = ActualSequentialSampler(self.train_idx[idxs_labeled])
			# ラベルありデータ用のデータローダーを作成
			tgts_loader = torch.utils.data.DataLoader(self.dset, sampler=tgts_sampler, num_workers=4, \
													batch_size=self.args.batch_size, drop_last=False)
			# ラベルありターゲットデータの埋め込み、ラベル、予測、ペナルティ項付き埋め込みを取得
			tgts_logits, tgts_lab, tgts_preds, tgts_pen_emb = utils.get_embedding(self.model, tgts_loader, self.device, self.num_classes, \
																		self.args, with_emb=True, emb_dim=emb_dim)

			# ソース+ターゲットラベルありデータの真のラベルを更新
			ST_y = torch.cat([ST_y, tgts_lab], dim=0)  # Ground truth of src+stgt 
			# ロジットを更新 (ターゲットラベルなし + ソース + ターゲットラベルあり)
			UST_logits = torch.cat([UST_logits, tgts_logits], dim=0)  # tgtuns + src + tgts 's logits 
			# 予測ラベルを更新
			UST_plab = torch.cat([UST_plab, tgts_preds]) 
			# 真のラベルを更新
			UST_label = torch.cat([UST_label, tgts_lab], dim=0) 
		else:
			tgts_loader = None
		# NumPy配列に変換 (CPU)
		ST_y = ST_y.cpu().numpy()
		UST_plab = UST_plab.cpu().numpy() 

		# ターゲットラベルなしデータの真のラベルを-1で初期化、その後ソースとターゲットラベルありの真のラベルを結合
		UST_y = np.concatenate([-1*np.ones(len(idxs_unlabeled)), ST_y])
		# ロジットから確率を計算
		UST_prob = F.softmax(UST_logits, dim=1)  # N,C 
		# 各サンプルの最大確率 (確信度) を取得
		UST_conf = UST_prob.max(dim=1)[0].cpu().numpy()  # unlab_lab_confidence
		# 確信度がしきい値以上のサンプルのインデックスを取得
		conf_index = np.where(UST_conf > self.args.sele_conf_thred)[0]   # model_conf > \tau  conf_index
		# ソースとターゲットラベルありデータの中で確信度がしきい値以上のサンプルのインデックスを取得
		ST_conf_index = conf_index[conf_index > len(idxs_unlabeled)] 
		# ソースとターゲットラベルありデータの中で確信度がしきい値以下のサンプルのインデックスを取得
		ST_uncertain_index = np.setdiff1d(np.arange(len(idxs_unlabeled), len(UST_y)), ST_conf_index)   # ST_uncertain_index
		# 損失を計算
		UST_loss = every_loss(UST_logits, UST_label.long()) 

		# 確信度が高く、予測が間違っているサンプルの損失
		STci_loss.extend(np.array(UST_loss)[ST_conf_index][UST_plab[ST_conf_index] != UST_y[ST_conf_index]])
		# 確信度が高く、予測が正しいサンプルの損失
		STcc_loss.extend(np.array(UST_loss)[ST_conf_index][UST_plab[ST_conf_index] == UST_y[ST_conf_index]])
		# 確信度が低く、予測が間違っているサンプルの損失
		STui_loss.extend(np.array(UST_loss)[ST_uncertain_index][UST_plab[ST_uncertain_index] != UST_y[ST_uncertain_index]]) 
		# 確信度が低く、予測が正しいサンプルの損失
		STuc_loss.extend(np.array(UST_loss)[ST_uncertain_index][UST_plab[ST_uncertain_index] == UST_y[ST_uncertain_index]]) # clean pseudo label in Q

		# ソースとターゲットラベルありデータの損失
		ST_gt_loss = every_loss(UST_logits[len(idxs_unlabeled):], UST_label[len(idxs_unlabeled):].cpu().long())
  		# 類似度モデルを用いてターゲットラベルなしデータの予測ラベルを計算
		tgtuns_topkLabel = self.calpro_fixpred(src_lab, src_pen_emb, tgtuns_logits, tgtuns_pen_emb, self.args.k_feat, tgts_lab, tgts_pen_emb)  

		# 必要なライブラリのインポート
		import pandas as pd
		from sklearn.metrics import accuracy_score
		# ここに正解率を計算するコードを追加
		# ターゲットの未ラベルデータの真のラベルと予測ラベルをCPU上のNumPy配列に変換
		tgtuns_true = tgtuns_lab.cpu().numpy()
		tgtuns_pred = tgtuns_topkLabel.cpu().numpy()

		# 全体の正解率を計算
		tgtuns_accuracy = np.mean(tgtuns_true == tgtuns_pred) * 100  # パーセンテージ表示

		# 各クラスごとの正解率を計算
		class_accuracies_tgtuns = {}
		for cls in range(self.num_classes):
			# クラスclsに属するサンプルのインデックスを取得
			idxs = np.where(tgtuns_true == cls)[0]
			if len(idxs) > 0:
				cls_acc = accuracy_score(tgtuns_true[idxs], tgtuns_pred[idxs]) * 100
				class_accuracies_tgtuns[cls] = cls_acc
			else:
				class_accuracies_tgtuns[cls] = None  # サンプルが存在しない場合

		# データフレームにまとめて表示
		overall_accuracy = {'Accuracy (%)': [tgtuns_accuracy]}
		df_overall = pd.DataFrame(overall_accuracy, index=['Overall'])

		class_accuracy = {'Class': [], 'Accuracy (%)': []}
		for cls in range(self.num_classes):
			if class_accuracies_tgtuns[cls] is not None:
				class_accuracy['Class'].append(cls)
				class_accuracy['Accuracy (%)'].append(class_accuracies_tgtuns[cls])
			else:
				class_accuracy['Class'].append(cls)
				class_accuracy['Accuracy (%)'].append('No samples')

		df_class = pd.DataFrame(class_accuracy)
		df_results = pd.concat([df_overall, df_class.set_index('Class')])

		#print("Similarity Model Accuracy on Target Unlabeled Data:")
		#print(df_results)

		# 結果をCSVファイルに保存（必要に応じて）
		#df_results.to_csv('similarity_model_accuracy.csv', index=True)

		##############################################################################
		##############################################################################
		##############################################################################
		src_topkLabel = self.calpro_fixpred(src_lab, src_pen_emb, src_logits, src_pen_emb, self.args.k_feat, tgts_lab, tgts_pen_emb)  

		# 必要なライブラリのインポート
		import pandas as pd
		from sklearn.metrics import accuracy_score
		# ここに正解率を計算するコードを追加
		# ターゲットの未ラベルデータの真のラベルと予測ラベルをCPU上のNumPy配列に変換
		src_true = src_lab.cpu().numpy()
		src_pred = src_topkLabel.cpu().numpy()

		# 全体の正解率を計算
		src_accuracy = np.mean(src_true == src_pred) * 100  # パーセンテージ表示

		# 各クラスごとの正解率を計算
		class_accuracies_src = {}
		for cls in range(self.num_classes):
			# クラスclsに属するサンプルのインデックスを取得
			idxs = np.where(src_true == cls)[0]
			if len(idxs) > 0:
				cls_acc = accuracy_score(src_true[idxs], src_pred[idxs]) * 100
				class_accuracies_src[cls] = cls_acc
			else:
				class_accuracies_src[cls] = None  # サンプルが存在しない場合

		# データフレームにまとめて表示
		overall_accuracy = {'Accuracy (%)': [src_accuracy]}
		df_overall = pd.DataFrame(overall_accuracy, index=['Overall'])

		class_accuracy = {'Class': [], 'Accuracy (%)': []}
		for cls in range(self.num_classes):
			if class_accuracies_src[cls] is not None:
				class_accuracy['Class'].append(cls)
				class_accuracy['Accuracy (%)'].append(class_accuracies_src[cls])
			else:
				class_accuracy['Class'].append(cls)
				class_accuracy['Accuracy (%)'].append('No samples')

		df_class = pd.DataFrame(class_accuracy)
		df_results = pd.concat([df_overall, df_class.set_index('Class')])

		##############################################################################
		##############################################################################
		##############################################################################

		# ターゲットラベルなしデータの損失を更新 (類似度モデルの予測ラベルを使用)
		adapt_ploss = every_loss(tgtuns_logits, tgtuns_topkLabel.long())
		UST_loss = torch.cat([adapt_ploss, ST_gt_loss])		
		U_loss.extend(np.array(UST_loss)[np.arange(0, len(idxs_unlabeled))])
		loss_lab.extend(np.array(UST_loss)[len(idxs_unlabeled):])

		# GMM学習のための正規化
		UST_loss = np.array(UST_loss).reshape(-1)
		max_lossItem = max(UST_loss) # max(loss_assist_ALL) 
		min_lossItem = min(UST_loss) # min(loss_assist_ALL)
		
		STci_loss = (np.array(STci_loss) - min_lossItem) / (max_lossItem - min_lossItem)
		STcc_loss = (np.array(STcc_loss) - min_lossItem) / (max_lossItem - min_lossItem)
		STui_loss = (np.array(STui_loss) - min_lossItem) / (max_lossItem - min_lossItem)
		STuc_loss = (np.array(STuc_loss) - min_lossItem) / (max_lossItem - min_lossItem)
		U_loss = (np.array(U_loss) - min_lossItem) / (max_lossItem - min_lossItem)  # 正規化されたラベルなしデータの損失

		# STui_lossが空の場合の例外処理: STci_lossから確信度が最も低いサンプルを選択し、STui_lossに追加
		# STui_lossが空でSTci_lossが空でない場合、STci_lossの中で確信度が最も低いサンプルをSTui_lossに移動する
		if len(STui_loss) == 0 and len(STci_loss) > 0:
			min_conf_idx = np.argmin(UST_conf[ST_conf_index][UST_plab[ST_conf_index] != UST_y[ST_conf_index]])  # STci_lossの中で最も確信度が低いサンプルのインデックス
			STui_loss = np.append(STui_loss, STci_loss[min_conf_idx]) # STui_lossに移動
			STci_loss = np.delete(STci_loss, min_conf_idx) # STci_lossから削除

		# STci_lossが空の場合: STui_lossから信頼度が最も高いサンプルを選択し、該当するSTci_lossに追加
		if len(STci_loss) == 0 and len(STui_loss) > 0:
			# STui_lossの中で最も確信度が高いサンプルのインデックスを取得
			max_conf_idx = np.argmax(UST_conf[ST_uncertain_index][UST_plab[ST_uncertain_index] != UST_y[ST_uncertain_index]])  
			STci_loss = np.append(STci_loss, STui_loss[max_conf_idx]) # STci_lossに追加
			STui_loss = np.delete(STui_loss, max_conf_idx) # STui_lossから削除

		# STuc_lossが空の場合: STcc_lossから信頼度が最も低いサンプルを選択し、該当するSTuc_lossに追加
		if len(STuc_loss) == 0 and len(STcc_loss) > 0:
			# STcc_lossの中で最も確信度が低いサンプルのインデックスを取得
			min_conf_idx = np.argmin(UST_conf[ST_conf_index][UST_plab[ST_conf_index] == UST_y[ST_conf_index]])  
			STuc_loss = np.append(STuc_loss, STcc_loss[min_conf_idx]) # STuc_lossに追加
			STcc_loss = np.delete(STcc_loss, min_conf_idx) # STcc_lossから削除

		# STcc_lossが空の場合: STuc_lossから信頼度が最も高いサンプルを選択し、該当するSTcc_lossに追加
		if len(STcc_loss) == 0 and len(STuc_loss) > 0:
			# STuc_lossの中で最も確信度が高いサンプルのインデックスを取得
			max_conf_idx = np.argmax(UST_conf[ST_uncertain_index][UST_plab[ST_uncertain_index] == UST_y[ST_uncertain_index]])  
			STcc_loss = np.append(STcc_loss, STuc_loss[max_conf_idx]) # STcc_lossに追加
			STuc_loss = np.delete(STuc_loss, max_conf_idx) # STuc_lossから削除


		# ラベルありデータの特徴量とラベルを作成
		x_labeled = np.concatenate([STcc_loss, STci_loss, STuc_loss, STui_loss])  # [3,2,1,0]
		y_labeled = np.concatenate([3*np.ones(len(STcc_loss)), 2*np.ones(len(STci_loss)), np.ones(len(STuc_loss)), np.zeros(len(STui_loss))])  # STcc_loss
		# ラベルなしデータの特徴量を作成
		x_unlabeled = copy.deepcopy(U_loss)   

		s_time = time()
		# 半教師ありガウス混合モデルを初期化
		m_ssGaussianMixture = ss_GaussianMixture()
		# 半教師ありガウス混合モデルを学習
		ss_GMM_parameter = m_ssGaussianMixture.fit(x_labeled.reshape(-1,1), y_labeled, x_unlabeled.reshape(-1,1), beta = 0.50, tol = 0.1, max_iterations = 20, early_stop = 'True')   
		gmm_save_dir = os.path.join(self.exp_path, 'gmm')
		os.makedirs(gmm_save_dir, exist_ok=True)
		gmm_save_file = os.path.join(gmm_save_dir, f"round_{self.query_count}.pkl")
		with open(gmm_save_file, "wb") as f:
			pickle.dump(m_ssGaussianMixture, f)
		# 学習済みGMMモデルとパラメータを保存
		self.GMM_models['GMM_model'] = {'ssGMM_Parameter': ss_GMM_parameter, 
										'min_loss': min_lossItem, 
										'max_loss': max_lossItem
											}

		# ラベルごとのサンプル数を出力
		"""
		print("y_labeled")
		unique_elements, counts = np.unique(y_labeled, return_counts=True)
		for element, count in zip(unique_elements, counts):
			print(f"{element}: {count}")
		"""
		
		# 学習済みGMMモデルでラベルなしデータの確率を予測
		ssGMM_i = m_ssGaussianMixture
		unlab_GMMprobs = ssGMM_i.predict(U_loss.reshape(-1,1), proba=True)  #[unstgt_num, 4] 
		# UIの確率を取得
		unlab_component_conf = np.max(unlab_GMMprobs[:,0:1], axis=1)
		# UCの確率を取得
		unlab_UC_conf = np.max(unlab_GMMprobs[:,1:2], axis=1)
		# 各サンプルの予測カテゴリを取得
		predicted_categories = np.argmax(unlab_GMMprobs, axis=1)  # [未ラベルデータ数]
		# カテゴリごとのサンプルのインデックスを取得
		ui_indices = np.arange(len(idxs_unlabeled))[predicted_categories == 0]
		print(len(ui_indices))
		uc_indices = np.arange(len(idxs_unlabeled))[predicted_categories == 1]
		ci_indices = np.arange(len(idxs_unlabeled))[predicted_categories == 2]
		cc_indices = np.arange(len(idxs_unlabeled))[predicted_categories == 3]
		print(f"UI (Uncertain-Inconsistent): {len(ui_indices)} samples")
		print(f"UC (Uncertain-Consistent): {len(uc_indices)} samples")
		print(f"CI (Confident-Inconsistent): {len(ci_indices)} samples")
		print(f"CC (Confident-Consistent): {len(cc_indices)} samples")

		#サンプル選択
		if (self.args.sampling == "ui"):
			idx_in_unstgt = unlab_component_conf.argsort()[::-1][:n]   # select top-n items in descending order
		elif "all_ui" in self.args.sampling:
			idx_in_unstgt = unlab_component_conf.argsort()[::-1][:n]   # select top-n items in descending order
		elif "distance_and_ui_weighted" in self.args.sampling:
				def parse_string(input_str):
					parts = input_str.split("_")
					a = "_".join(parts[:-1])  # 数値以外の部分を結合
					b = float(parts[-1])      # 最後の部分を数値と仮定
					return a, b
				a, b = parse_string(self.args.sampling)
				weights = b
				idx_in_unstgt = active_learning.distance_and_ui_weighted(ui_indices, cc_indices, unlab_component_conf, tgtuns_pen_emb, tgtuns_preds, n, tgts_loader, tgtuns_logits, tgtuns_lab, unlab_UC_conf,weights)	
		elif self.args.sampling == "distance_and_ui_curriculum_weighted":
				idx_in_unstgt = active_learning.distance_and_ui_curriculum_weighted(ui_indices, cc_indices, unlab_component_conf, tgtuns_pen_emb, tgtuns_preds, n, tgts_loader, tgtuns_logits, tgtuns_lab, unlab_UC_conf,self.query_count)	
		elif "distance_and_ui_weighted_source_minority" in self.args.sampling:
				def parse_string(input_str):
					parts = input_str.split("_")
					a = "_".join(parts[:-1])  # 数値以外の部分を結合
					b = float(parts[-1])      # 最後の部分を数値と仮定
					return a, b
				a, b = parse_string(self.args.sampling)
				weights = b
				idx_in_unstgt = active_learning.distance_and_ui_weighted_source_minority(ui_indices, cc_indices, unlab_component_conf, tgtuns_pen_emb, tgtuns_preds, n, tgts_loader, tgtuns_logits, tgtuns_lab, unlab_UC_conf,weights,src_lab)	
		elif self.args.sampling == "margin":
				idx_in_unstgt = active_learning.margin(ui_indices, cc_indices, unlab_component_conf, tgtuns_pen_emb, tgtuns_preds, n, tgts_loader, tgtuns_logits, tgtuns_lab, unlab_UC_conf)	
		elif self.args.sampling == "entropy":
				idx_in_unstgt = active_learning.entropy(ui_indices, cc_indices, unlab_component_conf, tgtuns_pen_emb, tgtuns_preds, n, tgts_loader, tgtuns_logits, tgtuns_lab, unlab_UC_conf)			
		elif "random" in self.args.sampling:
				idx_in_unstgt = active_learning.random(ui_indices, cc_indices, unlab_component_conf, tgtuns_pen_emb, tgtuns_preds, n, tgts_loader, tgtuns_logits, tgtuns_lab, unlab_UC_conf)
		else:
			print("sampling strategy not defined!!!")
  
  		# 選択されたサンプルのインデックス (ターゲット訓練セット内)
		selected_idxs = np.array(idxs_unlabeled[idx_in_unstgt])   # index in target train set
		print(unlab_component_conf.shape)
		print(len(idx_in_unstgt),idx_in_unstgt.shape, idx_in_unstgt)
		print(len(idxs_unlabeled),idxs_unlabeled.shape)
		print(len(selected_idxs),selected_idxs.shape, selected_idxs)
		########################################################################################################
		# 選択された画像のパスを保存
		selected_indices_in_dataset = self.train_idx[selected_idxs]

		# 画像パスを取得 (データセットに画像パスが格納されていない場合はロード)
		if self.dset.data is None:
			self.dset.data, self.dset.labels = make_dataset_fromlist(self.dset.txt_file)

		selected_image_paths = self.dset.data[selected_indices_in_dataset]

		# 画像パスを保存 (ラウンドごとにファイルを作成)
		output_dir = os.path.join(self.exp_path, 'selected_image')
		os.makedirs(output_dir, exist_ok=True)  # フォルダが存在しなければ作成
		output_file = os.path.join(output_dir, 'round_{}.txt'.format(self.query_count))
		with open(output_file, 'w') as f:
			for path in selected_image_paths:
				f.write(path + '\n')

		print("ラウンド{}で選択されたサンプルの画像パスを保存しました：{}".format(self.query_count, output_file))
		########################################################################################################
		# 1. CCカテゴリの予測結果からマイノリティクラスを特定
		cc_preds = tgtuns_preds[cc_indices].numpy().astype(int)  # 浮動小数点型を整数型に変換
		cc_class_counts = np.bincount(cc_preds, minlength=50)  # クラス数は50と仮定
		num_minority = len(cc_class_counts) // 2  # 少数クラスをクラスの半分と定義
		cc_minority_class = np.argsort(cc_class_counts)[:num_minority]  # 出現頻度が少ないクラスを取得
    
		idx_lab = np.arange(len(self.train_idx))[self.idxs_lb]
		tgtuns_mconfs = F.softmax(tgtuns_logits, dim=1).max(dim=1)[0].reshape(-1)
		min_num, min_num_gmm1 = (self.query_count + 1) * n, (self.query_count + 1) * n

		# GMMのコンポーネント3に基づいてデータローダーを作成
		cc_loader, idxs3_in_unstgt = self.get_gmm_conf_loader_noactive(tgtuns_logits, idxs_unlabeled, unlab_GMMprobs, min_num, idx_in_unstgt, cc_minority_class,compnent=3)
		# GMMのコンポーネント1に基づいてデータローダーを作成
		uc_loader, idxs1_in_unstgt = self.get_gmm_conf_loader_noactive(tgtuns_logits, idxs_unlabeled, unlab_GMMprobs, min_num_gmm1, idx_in_unstgt, cc_minority_class,compnent=1)

		print('-----self.query_count, min_num',self.query_count, min_num)

		all_cc_loader = self.get_gmm_loader_all(cc_indices, compnent=3)
		all_uc_loader = self.get_gmm_loader_all(uc_indices, compnent=1)
		#all_ci_loader = self.get_gmm_loader_all(ci_indices, compnent=2)

		return selected_idxs, cc_loader, uc_loader
	
	# If the amount of component is smaller than min_num, then select min_num samples according to confidence;
	# If it is larger than min_num, then select all the items in this component
	def get_gmm_conf_loader_noactive(self, U_logits, idxs_unlabeled, gmm_probs, min_num, sele_idxs_in_U, cc_minority_class, compnent=3):
		U_confs_all = gmm_probs[:,compnent].reshape(-1)

		unsele_idxs_in_U = np.setdiff1d(np.arange(len(idxs_unlabeled)), sele_idxs_in_U)
		U_logits_unsele, U_confs_unsele = U_logits[unsele_idxs_in_U], U_confs_all[unsele_idxs_in_U]

		if "_distance_and_ui" or "all_random" in self.args.sampling:
			conf_idx_in_unsele = utils.get_conf_balance_for_subset_majority_at_least_one(U_logits_unsele, U_confs_unsele, min_num, self.num_classes, cc_minority_class)
		conf_idx_in_U = unsele_idxs_in_U[conf_idx_in_unsele] 
		U_conf_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled[conf_idx_in_U]]) 
		fm_dataset = copy.deepcopy(self.dset) 
		if compnent == 3: fm_dataset.with_strong = True 
		U_conf_loader = torch.utils.data.DataLoader(fm_dataset, sampler=U_conf_sampler, num_workers=self.args.num_workers, \
												batch_size=self.args.batch_size, drop_last=False)
		return U_conf_loader, conf_idx_in_U 
	
	def get_gmm_loader_all(self, component_idx, compnent=3):
		U_conf_sampler = ActualSequentialSampler(self.train_idx[component_idx])
		fm_dataset = copy.deepcopy(self.dset) 
		if compnent == 3: fm_dataset.with_strong = True 
		component_loader = torch.utils.data.DataLoader(fm_dataset, sampler=U_conf_sampler, num_workers=self.args.num_workers, \
    								batch_size=self.args.batch_size, drop_last=False)
		return component_loader