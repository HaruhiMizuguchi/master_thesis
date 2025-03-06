# -*- coding: utf-8 -*-

# 標準ライブラリのインポート
import argparse
import os
import random
import time
import pickle
from collections import defaultdict, Counter

# サードパーティライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import torch
from scipy.stats import pearsonr
import scipy.stats
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import japanize_matplotlib

# ローカルライブラリのインポート
from data import ASDADataset
import utils as utils
from adapt.models.models import get_model
from adapt.models.task_net import *

# シード値の設定
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

SCOPES = ['https://www.googleapis.com/auth/drive']
# Google APIの認証情報を含むJSONファイルのパス
# OAuth 2.0 クライアントシークレットファイル（例: credentials.json）
CLIENT_SECRET_FILE = "--apps.googleusercontent.com.json"
# トークンファイルのパス
TOKEN_FILE = 'token.json'
# Google DriveのルートフォルダID
DRIVE_ROOT_FOLDER_ID = '' 

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# その他のインポート設定
import importlib
import import_ipynb
import analyze_utils
#importlib.reload(analyze_utils)

def read_gmm(round_num, args):
    gmm_dir = os.path.join(
        'with_gmm_only_tgt_sup_exp_record_6r_150b_fix_seed_not_augment', 'multi-round', "domainnet_50",
        f'{args.source_domain}2{args.target_domain}',
        f'source_GMM_self_ft_{int(args.num_rounds)}r_{int(args.total_budget)}b_{args.cnn}',
        'warmupft-adapt_lr1e-05-wd1e-05-srcw0.1-ccw0.5-ucw0.1', f'subset_{args.subset_idx}', f'{args.sampling}',
        f'{args.run_num}', 'gmm'
    )
    gmm_path = os.path.join(gmm_dir, f"round_{round_num+1}.pkl")
    
    # .pklファイルを読み込み
    with open(gmm_path, "rb") as f:
        loaded_ss_GMM_parameter = pickle.load(f)
    
    # リストの場合、リスト内のGaussianMixtureインスタンスを返す
    if isinstance(loaded_ss_GMM_parameter, list):
        for item in loaded_ss_GMM_parameter:
            # インスタンスがGaussianMixtureか確認
            if hasattr(item, "predict"):
                return item  # GaussianMixtureインスタンスを返す
        raise ValueError("リスト内にGaussianMixtureインスタンスが見つかりませんでした。")
    else:
        # すでにGaussianMixtureインスタンスの場合、そのまま返す
        return loaded_ss_GMM_parameter

# Calculate category-wise centroids
def calpro_fixpred(src_lab,  src_pen_emb, tgtuns_logits , tgtuns_pen_emb, k_feat, tgts_lab=[], tgts_pen_emb=[], \
                   num_classes = 50, emb_dim = 512):

    cls_prototypes = torch.zeros([num_classes, emb_dim])
    tgtuns_preds = torch.argmax(tgtuns_logits, dim=1)
    for i in range(num_classes):
        anchor_i = src_pen_emb[src_lab == i]
        if tgts_lab is not None:
            emb = tgts_pen_emb[tgts_lab == i] 
            if len(emb) > 0: anchor_i = torch.cat([anchor_i, emb],dim=0)
        anchor_i = anchor_i.mean(dim=0).reshape(-1)
        cls_prototypes[i,:] = anchor_i

    fixed_unstgt_preds = utils.utils.topk_feat_pred(tgtuns_logits, tgtuns_pen_emb, cls_prototypes, k_feat= k_feat, k_pred=num_classes)
    return fixed_unstgt_preds

def predict_category(round_num, args, source_e_features,source_logits,source_labels,unlabeled_target_e_features, unlabeled_target_logits,\
                  labeled_target_e_features,labeled_target_logits,labeled_target_labels):
    k_feat = 32
    unlabeled_target_topkLabel = calpro_fixpred(source_labels, source_e_features, unlabeled_target_logits, \
                                            unlabeled_target_e_features, k_feat, labeled_target_labels, labeled_target_e_features)
    gmm = read_gmm(round_num, args)
    every_loss =  nn.CrossEntropyLoss(reduction="none")
    adapt_ploss = every_loss(unlabeled_target_logits, unlabeled_target_topkLabel.long().to(unlabeled_target_logits.device))
    source_loss = every_loss(source_logits,source_labels)
    if labeled_target_e_features is None:
        labeled_target_loss = None
        UST_loss = torch.cat([adapt_ploss, source_loss],dim=0)
    else:
        labeled_target_loss = every_loss(labeled_target_logits, labeled_target_labels)
        UST_loss = torch.cat([adapt_ploss, source_loss, labeled_target_loss],dim=0)
    UST_loss = np.array(UST_loss.cpu()).reshape(-1)
    max_lossItem = max(UST_loss) # max(loss_assist_ALL) 
    min_lossItem = min(UST_loss) # min(loss_assist_ALL)
    adapt_ploss = (np.array(adapt_ploss.cpu()) - min_lossItem) / (max_lossItem - min_lossItem)  # 正規化されたラベルなしデータの損失
    source_loss = (np.array(source_loss.cpu()) - min_lossItem) / (max_lossItem - min_lossItem)
    if labeled_target_loss is not None:
        labeled_target_loss = (np.array(labeled_target_loss.cpu()) - min_lossItem) / (max_lossItem - min_lossItem)
    
    unlab_GMMprobs = gmm.predict(adapt_ploss.reshape(-1,1), proba=True)  #[unstgt_num, 4]

    # unlab_GMMprobsから各データポイントの予測カテゴリを取得
    predicted_categories = np.argmax(unlab_GMMprobs, axis=1)

    return unlab_GMMprobs, adapt_ploss, source_loss, labeled_target_loss

def calc_category(round_num, data_dict, args):
    print("read model")
    #model = analyze_utils.read_model(round_num, args)
    print("read model dirve")
    model = analyze_utils.read_model_drive(round_num, args)
    source_train_loader = data_dict['src_train_loader']
    unlabeled_target_loader = data_dict['unlabeled_target_loader_dict'][f"{round_num}"]
    labeled_target_loader = data_dict['labeled_target_loader_dict'][f"{round_num}"]
    print("compute source")
    source_train_e_features, source_train_logits, source_train_labels, source_train_paths = analyze_utils.ComputeFeatureAndPreds(source_train_loader, model)
    print("compute unlabeled target")
    unlabeled_target_e_features, unlabeled_target_logits, unlabeled_target_labels, unlabeled_target_paths = analyze_utils.ComputeFeatureAndPreds(unlabeled_target_loader, model)
    print("compute labeled target")
    if len(labeled_target_loader) > 0:
        labeled_target_e_features, labeled_target_logits, labeled_target_labels, labeled_target_paths = analyze_utils.ComputeFeatureAndPreds(labeled_target_loader, model)
    else:
        labeled_target_e_features, labeled_target_logits, labeled_target_labels, labeled_target_paths = None, None, None, None
    
    unlab_GMMprobs, adapt_ploss, source_loss, labeled_target_loss= predict_category(round_num, args,source_train_e_features, source_train_logits, source_train_labels,\
                                     unlabeled_target_e_features, unlabeled_target_logits,labeled_target_e_features, labeled_target_logits, labeled_target_labels)
    data_dict_when_calc_category = {
        "source_train_e_features": source_train_e_features,
        "source_train_logits": source_train_logits,
        "source_train_labels": source_train_labels,
        "source_train_paths": source_train_paths,
        "unlabeled_target_e_features": unlabeled_target_e_features,
        "unlabeled_target_logits": unlabeled_target_logits,
        "unlabeled_target_labels": unlabeled_target_labels,
        "unlabeled_target_paths": unlabeled_target_paths,
        "labeled_target_e_features": labeled_target_e_features,
        "labeled_target_logits": labeled_target_logits,
        "labeled_target_labels": labeled_target_labels,
        "labeled_target_paths": labeled_target_paths,
        "source_loss": source_loss,
        "adapt_ploss": adapt_ploss,
        "labaled_target_loss": labeled_target_loss,
        "unlab_GMMprobs": unlab_GMMprobs
    }
    return unlab_GMMprobs, data_dict_when_calc_category

def main():
    parser = argparse.ArgumentParser(description="Process domain adaptation arguments")
    parser.add_argument("--domain_idxs", type=int, nargs='+', required=True, help="Domain indices")
    parser.add_argument("--subset_idxs", type=int, nargs='+', required=True, help="Subset indices")
    parser.add_argument("--samplings", type=str, nargs='+', required=True, help="Sampling strategies")
    args = parser.parse_args()

    args.source_domains = ["clipart", "real", "sketch", "clipart", "real"]
    args.target_domains = ["sketch", "clipart", "painting", "quickdraw", "sketch"]
    args.cnn = "ResNet34"
    args.dataset = "domainnet_50"
    args.num_classes = 50
    args.total_budget = 150
    args.num_rounds = 6
    args.run_num = 1

    all_data_dict = torch.load("result_analyze_label_distribution/all_data_dict.pt", map_location="cpu")
    all_data_dict_when_calc_category = torch.load("result_analyze_label_distribution/all_data_dict_when_calc_category.pt", map_location="cpu")

    all_data_dict = analyze_utils.compute_all_data_dict(args, all_data_dict)

    for domain_idx in args.domain_idxs:
        args.source_domain = args.source_domains[domain_idx]
        args.target_domain = args.target_domains[domain_idx]
        for subset_idx in args.subset_idxs:
            args.subset_idx = subset_idx
            for sampling in args.samplings:
                args.sampling = sampling
                args.domain_key = f"{args.source_domain}2{args.target_domain}"
                args.subset_key = f"subset_{args.subset_idx}"
                args.sampling_key = f"{args.sampling}"
                for round_num in range(args.num_rounds):
                    data_dict = all_data_dict[args.domain_key][args.subset_idx][args.sampling]
                    unlab_GMMprobs, data_dict_when_calc_category = calc_category(round_num, data_dict, args)
                    all_data_dict_when_calc_category.setdefault(args.domain_key, {}).setdefault(args.subset_idx, {}).setdefault(args.sampling, {}).\
                        setdefault(round_num, data_dict_when_calc_category)
                
    torch.save(all_data_dict, "result_analyze_label_distribution/all_data_dict.pt")
    torch.save(all_data_dict_when_calc_category, "result_analyze_label_distribution/all_data_dict_when_calc_category.pt")

if __name__ == "__main__":
    main()