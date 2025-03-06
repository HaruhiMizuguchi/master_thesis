# -*- coding: utf-8 -*-
from math import ceil
import os
import random
import argparse
import shutil
from omegaconf import OmegaConf
import copy
import pprint
from collections import defaultdict
from tqdm import trange
from sklearn.metrics import recall_score

import numpy as np
import torch
import pandas as pd
from pdb import set_trace
import seed
random.seed(seed.a)
torch.manual_seed(seed.a)
np.random.seed(seed.a)
torch.cuda.manual_seed(seed.a)
torch.cuda.manual_seed_all(seed.a)  # CUDA全デバイスに対してシード設定
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from adapt.models.models import get_model
import utils as utils
from data import ASDADataset
#from sample import *
from sample import *
from sample_sota import get_sota_strategy
import pdb
import time

from adapt.models.task_net import *

import drive_utils as drive_utils
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from config.secrets import GOOGLE_CLIENT_SECRET_FILE, GOOGLE_TOKEN_FILE, PROJECT_BASE_PATH

SCOPES = ['https://www.googleapis.com/auth/drive']
# Replace hardcoded values with imported values
CLIENT_SECRET_FILE = GOOGLE_CLIENT_SECRET_FILE
TOKEN_FILE = GOOGLE_TOKEN_FILE
DRIVE_ROOT_FOLDER_ID = '1mLWoJ41KzOh37csNhccw6uC1p5F8u7l3'  # Google DriveのルートフォルダID
# 認証してサービスを構築
#creds = utils.authenticate_google_drive()
#service = build('drive', 'v3', credentials=creds)

def run_active_adaptation(args, source_model, src_dset, num_classes, device, writer, exp_path, src_start_test_acc, run_num, fe=None, clf=None):
    """
    Runs active domain adaptation experiments
    """
    # Initialize a list to store macro recalls for all rounds
    all_macro_recalls = []

    # Load source data
    src_train_loader, src_val_loader, src_test_loader, _  = src_dset.get_loaders(num_workers=args.num_workers) 
    
    # Load target data
    target_dset = ASDADataset(args.dataset, args.target, "target", subset_idx = args.subset_idx, valid_ratio=0, batch_size=args.batch_size)  

    ############################################################################
    #target_train_dset, _, _ = target_dset.get_dsets(apply_transforms=True)
    target_train_dset, _, _ = target_dset.get_dsets(apply_transforms=False)   
    
    target_train_loader, target_val_loader, target_test_loader, train_idx = target_dset.get_loaders(num_workers=args.num_workers) 

    print("len(target_train_dset):",len(target_train_dset))
    
    # Sample varying % of target data 
    if args.dataset in ['domainnet', 'office31', 'officehome', 'domainnet_50']:
        sampling_ratio = [(args.total_budget/args.num_rounds) * n for n in range(args.num_rounds+1)]   #0, 1*args.total_budget/args.num_rounds...  len args.num_rounds+1
    else: raise NotImplementedError

    # Evaluate source model on target test 
    transfer_perf, transfer_perf_topk = utils.utils.test(source_model, device, target_test_loader, topk=1) 
    out_str = '{}->{} performance (Before {}): Task={:.2f}  Topk Acc={:.2f}'.format(args.source, args.target, args.warm_strat, transfer_perf, transfer_perf_topk)
    print(out_str)
    

    print('------------------------------------------------------\n')
    print('Running strategy: Init={}, AL={}, Pretrain={}'.format(args.model_init, args.al_strat, args.warm_strat))
    print('\n------------------------------------------------------')	

    # Choose appropriate model initialization
    if args.model_init == 'scratch':
        model, src_model = get_model(args.cnn, num_cls=num_classes).to(device), model
    elif args.model_init == 'source':
        model, src_model = source_model, source_model

    # Run unsupervised DA at round 0, where applicable
    sub_model = None # 
    if args.da_strat in ["dann"]:
        sub_model = utils.utils.get_disc(num_classes)
    
    test_mode = "ori" 
    start_perf = transfer_perf	
    
    #################################################################
    # Main Active DA loop
    #################################################################
    target_accs = defaultdict(list)
    test_accs = np.zeros([args.runs+1 ,args.num_rounds+1])
    test_accs_gc = np.zeros([args.runs+1, args.num_rounds+1])
    test_accs_run = np.zeros([int(args.runs+1)])
    test_accs[:,0] = start_perf
    #################################################################
    src_accs = np.zeros([args.runs+1 ,args.num_rounds+1])
    src_accs_run = np.zeros([int(args.runs+1)])
    src_accs[:,0] = src_start_test_acc
    #################################################################
    tgt_test_num_sample_per_class = np.zeros(50)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(target_test_loader):
            data, target = data.to(device), target.to(device)
            for i in range(len(target)):
                label = target[i].item()
                tgt_test_num_sample_per_class[int(label)] += 1
    sorted_tgt_test_num_sample_per_class = np.argsort(tgt_test_num_sample_per_class)
    minority_class = sorted_tgt_test_num_sample_per_class[:25]
    args.minority_class = minority_class.tolist()
    print("minority_class:",minority_class)
    majority_class = sorted_tgt_test_num_sample_per_class[25:]
    args.majority_class = majority_class.tolist()
    minority_class_accs = np.zeros(args.num_rounds+1)
    majority_class_accs = np.zeros(args.num_rounds+1)
    per_class_accuracy, per_class_accuracy_array = utils.utils.test_per_class_accuracy(model, device, target_test_loader, mode=test_mode)
    minority_class_accs[0] = np.mean(per_class_accuracy_array[minority_class])
    majority_class_accs[0] = np.mean(per_class_accuracy_array[majority_class])
    #################################################################
    tqdm_run = trange(args.runs)
    for run in tqdm_run: # Run over multiple experimental runs
        tqdm_run.set_description('Run {}'.format(str(run)))
        tqdm_run.refresh()
        tqdm_rat = trange(len(sampling_ratio[1:]))  #len = args.num_rounds
        target_accs[0.0].append(start_perf)
        
        # Making a copy for current run
        curr_model = copy.deepcopy(model)

        # Keep track of labeled vs unlabeled data
        idxs_lb = np.zeros(len(train_idx), dtype=bool)

        # Instantiate active sampling strategy
        if args.al_strat == "GMM":
            sampling_strategy = get_strategy(args.al_strat, target_train_dset, train_idx, curr_model, sub_model, device, args, \
                                                        writer, run, exp_path)		
        else:
            sampling_strategy = get_sota_strategy(args.al_strat, target_train_dset, train_idx, curr_model, sub_model, device, args, \
                                                writer, run, exp_path)		
        

        run_query_t, run_train_t = 0, 0 
        for ix in tqdm_rat: # Iterate over Active DA rounds
            ratio = sampling_ratio[ix+1]  #1
            tqdm_rat.set_description('# Target labels={:d}'.format(int(ratio)))
            tqdm_rat.refresh()

            # Select instances via AL strategy
            print('\nSelecting instances...')
            query_stime = time.time()			

            if args.al_strat == "GMM":
                idxs, uns_tgt_conf_loader, uns_tgt_unconf_loader = \
                                    sampling_strategy.query(int(sampling_ratio[1]), src_train_loader)
            elif args.al_strat in ["Alpha"]:
                idxs = sampling_strategy.query(int(sampling_ratio[1]), src_train_loader)
            elif args.al_strat in ["uniform", "CLUE", "entropy", "BADGE"]:
                idxs = sampling_strategy.query(int(sampling_ratio[1]))
            else: raise Exception("Not supported AL")
            
            # Record query time
            round_query_t = (time.time() - query_stime)
            run_query_t += round_query_t
            print("query of this round takes {} mins, or {:.2f} secs)".format(round_query_t//60, round_query_t ))
            
            idxs_lb[idxs] = True
            sampling_strategy.update(idxs_lb)  # update sampling_strategy.idxs_lb = idxs_lb
            if args.shuffle_src:
                src_train_loader, _, _, _ = src_dset.get_loaders(num_workers=args.num_workers) 

            # Update model with new data via DA strategy
            round_train_start = time.time()	
            if args.al_strat == "GMM": 
                best_model, qc_best_acc = sampling_strategy.train(target_train_dset, args, src_loader=src_train_loader, 
                                                    tgt_conf_loader=uns_tgt_conf_loader, tgt_unconf_loader=uns_tgt_unconf_loader)
            else:
                best_model, qc_best_acc = sampling_strategy.train(target_train_dset, args, src_loader=src_train_loader)
            
            best_da_model = copy.deepcopy(best_model)
            # チェックポイントを保存する前にディレクトリを作成
            #da_model_dir = os.path.join(f'models_fix_seed_{args.cnn}', args.dataset, f"{args.source}2{args.target}", f"{args.subset_idx}s_{int(args.total_budget)}b_{args.num_rounds}r",args.sampling,f"{run_num}")
            #da_model_dir = os.path.join(f'only_tgt_sup_models_fix_seed_not_augment_{args.cnn}', args.dataset, f"{args.source}2{args.target}", f"{args.subset_idx}s_{int(args.total_budget)}b_{args.num_rounds}r",args.sampling,f"{run_num}")
            da_model_dir = os.path.join(f'with_gmm_only_tgt_sup_models_fix_seed_not_augment_{args.cnn}', args.dataset, f"{args.source}2{args.target}", f"{args.subset_idx}s_{int(args.total_budget)}b_{args.num_rounds}r",args.sampling,f"{run_num}")
            #da_model_dir = os.path.join(f'models_{args.iter_num}_{args.cnn}', args.dataset, f"{args.source}2{args.target}", f"{args.subset_idx}s_{int(args.total_budget)}b_{args.num_rounds}r")
            os.makedirs(da_model_dir, exist_ok=True)
            # モデルを保存
            #torch.save(best_da_model.state_dict(), os.path.join(da_model_dir, f"round_{ix+1}.pth"))
            #ネットにつながるときの保存
            ####utils.save_model_to_drive(service, best_da_model.state_dict(), da_model_dir, f"round_{ix+1}.pth", DRIVE_ROOT_FOLDER_ID)
            #自分のpcを介した保存
            drive_folder = da_model_dir
            drive_file = f"round_{ix+1}.pth"
            drive_utils.save_model_drive_relay_pc(model,drive_folder,drive_file)
            
            round_train_t = time.time()-round_train_start
            run_train_t += round_train_t
            print("Training of this round takes {} mins, or {:.2f} secs)".format(round_train_t//60, round_train_t ))
            
            # Evaluate on target test and train splits
            test_perf, test_perf_topk = utils.utils.test(best_model, device, target_test_loader, mode=test_mode, topk=1)
            out_str = '{}->{} Test performance (Run {}  query_count {}, # Target labels={:d}): {:.2f}  Topk Acc={:.2f}'.format(args.source, args.target, run+1, \
                                                        sampling_strategy.query_count, int(ratio), test_perf, test_perf_topk)			

            writer.add_scalar('Run{}/TargetTestAcc'.format(run), test_perf,int(ratio))

            print(out_str)
            print('\n------------------------------------------------------\n')
            
            test_accs[run, ix+1] = test_perf
            ############################################################################################
            source_perf, source_perf_topk = utils.utils.test(best_model, device, src_test_loader, mode=test_mode, topk=1)
            src_accs[run, ix+1] = source_perf
            ############################################################################################
            target_accs[ratio].append(test_perf)
            ############################################################################################
            # ソースのテストデータに対して、各クラスの正解率を計算して保存
            src_per_class_accuracy, src_per_class_accuracy_array = utils.utils.test_per_class_accuracy(best_model, device, src_test_loader, mode=test_mode)
            #print("source Per-class Accuracy:", src_per_class_accuracy)
               # 結果をDataFrameに変換して保存
            src_per_class_accuracy_df = pd.DataFrame(list(src_per_class_accuracy.items()), columns=['Class', 'Accuracy (%)'])
            output_dir = os.path.join(exp_path,"src_per_class_accuracy_da_model")
            os.makedirs(output_dir,exist_ok=True)
            src_per_class_accuracy_csv_path = os.path.join(output_dir, f'round_{ix+1}.csv')
            src_per_class_accuracy_df.to_csv(src_per_class_accuracy_csv_path, index=False)
            print(f"Saved per-class accuracy to {src_per_class_accuracy_csv_path}")
            ############################################################################################
            ############################################################################################
            # 各クラスの正解率を計算
            #per_class_accuracy = test_per_class_accuracy(best_model, device, target_test_loader, mode=test_mode, class_index_to_name=class_index_to_name)
            per_class_accuracy, per_class_accuracy_array = utils.utils.test_per_class_accuracy(best_model, device, target_test_loader, mode=test_mode)
            minority_class_accs[ix+1] = np.mean(per_class_accuracy_array[minority_class])
            majority_class_accs[ix+1] = np.mean(per_class_accuracy_array[majority_class])
            #print("target Per-class Accuracy:", per_class_accuracy)

            # 結果をDataFrameに変換して保存
            per_class_accuracy_df = pd.DataFrame(list(per_class_accuracy.items()), columns=['Class', 'Accuracy (%)'])
            output_dir = os.path.join(exp_path,"per_class_accuracy")
            os.makedirs(output_dir,exist_ok=True)
            per_class_accuracy_csv_path = os.path.join(output_dir, f'round_{ix+1}.csv')
            per_class_accuracy_df.to_csv(per_class_accuracy_csv_path, index=False)
            print(f"Saved per-class accuracy to {per_class_accuracy_csv_path}")
            ##############################################################################################
            # Calculate macro recall
            y_true, y_pred = utils.utils.get_all_labels_and_predictions(best_model, device, target_test_loader)
            macro_recall = recall_score(y_true, y_pred, average='macro')
            print(f"Macro Recall for round {ix+1}: {macro_recall:.4f}")

            # Store macro recall for later
            all_macro_recalls.append(macro_recall)

        # Save all macro recalls to a single row CSV
        macro_recall_csv_path = os.path.join(exp_path, 'all_rounds_macro_recall.csv')
        with open(macro_recall_csv_path, 'w') as f:
            f.write(','.join([f'{x:.4f}' for x in all_macro_recalls]) + '\n')
        print(f"Saved all rounds macro recall to {macro_recall_csv_path}")

        
        run_qt_t = run_query_t + run_train_t
        print("----- Run-{} 	 	takes {}h-{}m ({} secs)".format(run, run_qt_t//3600, (run_qt_t%3600)//60, int(run_qt_t) ))
        print("----------- query 	takes {}h-{}m ({} secs)".format(run_query_t//3600, (run_query_t%3600)//60, int(run_query_t)) )
        print("----------- training takes {}h-{}m ({} secs)".format(run_train_t//3600, (run_train_t%3600)//60, int(run_train_t)) )

        test_accs_run[run] = max(qc_best_acc, test_perf)


        def make_serializable(obj):
            # OmegaConfのオブジェクトならシリアライズ可能な形式に変換
            if OmegaConf.is_config(obj):
                return OmegaConf.to_container(obj, resolve=True)
            
            # リストやタプルの場合、再帰的に変換
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            
            # 辞書の場合、再帰的に変換
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            
            # numpy.ndarray をリストに変換
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            
            # その他の型はそのまま返す
            else:
                return obj
            
        # 修正箇所
        # OmegaConf関連の型をPython標準の型に変換
        wargs = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
        target_accs['args'] = make_serializable(wargs)

        # データをJSONファイルに保存
        utils.utils.log(target_accs, os.path.join(exp_path, 'perf.json'))
        # Log at the end of every run
        #wargs = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
        #target_accs['args'] = wargs
        #utils.log(target_accs, os.path.join(exp_path,'perf.json') )
        
        if args.run_one:
            test_accs[1,:], test_accs[2,:] = test_accs[0,:], test_accs[0,:]
            test_accs_run[1:3] = max(qc_best_acc, test_perf)
            break

        if args.run_one:
            src_accs[1,:], src_accs[2,:] = src_accs[0,:], src_accs[0,:]
            src_accs_run[1:3] = max(qc_best_acc, source_perf)
            break

    test_accs[-1,:] = np.mean(test_accs[:int(args.runs),:],0)   # avg value
    accs_df = pd.DataFrame(test_accs, columns=sampling_ratio)
    accs_df.to_csv(os.path.join(exp_path,'test_accs.csv'),encoding='utf-8')
    
    src_accs[-1,:] = np.mean(src_accs[:int(args.runs),:],0)
    src_accs_df = pd.DataFrame(src_accs, columns=sampling_ratio)
    src_accs_df.to_csv(os.path.join(exp_path,'src_accs.csv'),encoding='utf-8')
    
    minority_class_accs_df = pd.DataFrame(minority_class_accs.reshape(1, -1), columns=sampling_ratio)
    minority_class_accs_df.to_csv(os.path.join(exp_path,'minority_class_accs.csv'),encoding='utf-8')
    
    majority_class_accs_df = pd.DataFrame(majority_class_accs.reshape(1, -1), columns=sampling_ratio)
    majority_class_accs_df.to_csv(os.path.join(exp_path,'majority_class_accs.csv'),encoding='utf-8')
 
 
    test_accs_run[-1] = np.max(test_accs_run[:3])  # max value
    test_accs_run = test_accs_run.reshape(1,-1)
    test_accs_run_df = pd.DataFrame(test_accs_run, columns=['run1','run2','run3','best'])
    test_accs_run_df.to_csv(os.path.join(exp_path,'test_accs_run.csv'),encoding='utf-8')
    return target_accs

def main():
    parser = argparse.ArgumentParser()
    # Experiment identifiers
    parser.add_argument('--id', type=str, help="Experiment identifier") # transfer pair
    parser.add_argument('-a', '--al_strat', type=str, help="Active learning strategy")
    parser.add_argument('-d', '--da_strat', type=str, default='ft', help="DA strat. Currently supports: {ft, self_ft}")   #during al
    parser.add_argument('--warm_strat', type=str, default='ft', help="DA strat. Currently supports: {ft}")   #warmup
    # Load existing configuration
    parser.add_argument('--gpu',type=str, default='0', help='which gpu to use') 
    parser.add_argument('--load_from_cfg', type=bool, default=True, help="Load from config?")
    parser.add_argument('--cfg_file', type=str, help="Experiment configuration file", default="config/domainnet/clipart2sketch.yml")
    
    parser.add_argument('--thread', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    #for fix pred
    parser.add_argument('--pro_type', type=str, default="lab_mean")
    parser.add_argument('--k_feat', type=int, help='k in top-k similarity')
    parser.add_argument('--run_one', type=bool, default=True)  # one random run only
    
    #for active learning
    parser.add_argument('--round_type', type=str, default='multi-round')
    parser.add_argument('--total_budget', type=float)
    parser.add_argument('--num_rounds', type=int)

    #for training model on source domain
    parser.add_argument('--cnn', type=str)
    parser.add_argument('--warmup_epochs', type=int)

    parser.add_argument('--lr', type=float)
    parser.add_argument('--wd', type=float)

    parser.add_argument('--ft_solve', type=str, default='solve') 
    parser.add_argument('--iter_rate', type=int, default=1)
    parser.add_argument('--model_init', type=str, default="source")
    parser.add_argument('--num_epochs', type=int, default=50)

    #for adaptation on target domain
    parser.add_argument('--adapt_lr', type=float, help="DA learning rate")	# 1e-5 for dn
    parser.add_argument('--iter_num', type=str)
    parser.add_argument('--adapt_num_epochs', type=int)
    parser.add_argument('--test_best', action='store_true') 
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--shuffle_src', type=bool, default=True) 
    
    parser.add_argument('--work_root', type=str, default="./") 
    parser.add_argument('--dataset', type=str, help="Dataset name (e.g., domainnet, office31, officehome)")
    parser.add_argument('--sampling', type=str, default='ui', help="sampling strategy(e.g., ui, cos_sim, euclid)")
    parser.add_argument('--subset_idx_argument', type=int, default=0)
    parser.add_argument('--loss', type=str, choices=['all', 'only_sup'], default='only_sup', help='Loss type to use. Choose "all" or "specific".')
    args_cmd = parser.parse_args()
    
    if args_cmd.load_from_cfg:
        args_cfg = dict(OmegaConf.load(args_cmd.cfg_file))
        args_cmd = vars(args_cmd)
        for k in args_cmd.keys():
            if args_cmd[k] is not None: args_cfg[k] = args_cmd[k]  # args_cmd as priority 
        args = OmegaConf.create(args_cfg)
    else: 
        args = args_cmd

    if args.subset_idx_argument != 0:
        args.subset_idx = args.subset_idx_argument
    
    #if not args.sampling in ["ui","cos_sim","euclid","cumulative_class_balance","per_round_class_balance","uncertainty_round_class_balance","uncertainty_not_balanced"]:
    #    print("sampling not defined")
    #    exit()
    pp = pprint.PrettyPrinter()
    pp.pprint(args)
    torch.set_num_threads(args.thread)
    os.environ["CUDA_VISIBLE_DEVICES"] = utils.utils.get_gpu_usedrate(1)[0]
    assert torch.cuda.is_available()
    device = torch.device(0) #use directly
    # Record
    writer, exp_path, run_num = utils.utils.gen_dir(args)
    print("current exp_path: \t", exp_path)
    

    # Load source data
    src_dset = ASDADataset(args.dataset, args.source, "source", subset_idx = args.subset_idx, batch_size=args.batch_size) 
    src_train_loader, src_val_loader, src_test_loader, _ = src_dset.get_loaders(num_workers=args.num_workers)  
    num_classes = src_dset.get_num_classes()
    print('Number of classes: {}'.format(num_classes))
    #da_model_dir = os.path.join(f'models_fix_seed_{args.cnn}', args.dataset, f"{args.source}2{args.target}", f"{args.subset_idx}s_{int(args.total_budget)}b_{args.num_rounds}r",args.sampling,f"{run_num}")
    #da_model_dir = os.path.join(f'only_tgt_sup_models_fix_seed_not_augment_{args.cnn}', args.dataset, f"{args.source}2{args.target}", f"{args.subset_idx}s_{int(args.total_budget)}b_{args.num_rounds}r",args.sampling,f"{run_num}")
    da_model_dir = os.path.join(f'with_gmm_only_tgt_sup_models_fix_seed_not_augment_{args.cnn}', args.dataset, f"{args.source}2{args.target}", f"{args.subset_idx}s_{int(args.total_budget)}b_{args.num_rounds}r",args.sampling,f"{run_num}")

    #da_model_dir = os.path.join(f'models_{args.iter_num}_{args.cnn}', args.dataset, f"{args.source}2{args.target}", f"{args.subset_idx}s_{int(args.total_budget)}b_{args.num_rounds}r")
    os.makedirs(da_model_dir, exist_ok=True)
    # Train/load a source model
    source_model = get_model(args.cnn, num_cls=num_classes).to(device)	
    if args.dataset == "domainnet_50":
        source_file = '{}_{}_{}_source.pth'.format(args.source, args.subset_idx, args.cnn)
    else:
        source_file = '{}_{}_source.pth'.format(args.source, args.cnn)
    source_path = os.path.join('checkpoints', 'source', args.dataset, source_file)	
    if os.path.exists(source_path): # Load existing source model
        print('Loading source checkpoint: {}'.format(source_path))
        source_model.load_state_dict(torch.load(source_path, map_location=device), strict=True)   #map location
        best_source_model = source_model
        if not os.path.exists(os.path.join(da_model_dir, "round_0.pth")):
            #torch.save(best_source_model.state_dict(), os.path.join(da_model_dir, f"round_0.pth"))
            #utils.save_model_to_drive(service, best_source_model.state_dict(), da_model_dir, f"round_0.pth", DRIVE_ROOT_FOLDER_ID)
            drive_folder = da_model_dir
            drive_file = f"round_0.pth"
            drive_utils.save_model_drive_relay_pc(best_source_model.state_dict(),drive_folder,drive_file)
    else:							# Train source model from scarach
        print('Training {} model...'.format(args.source))
        best_val_acc, best_source_model = 0.0, None
        source_optimizer = optim.Adam(source_model.parameters(), lr=args.lr, weight_decay=args.wd)

        for epoch in range(args.num_epochs):
            utils.utils.train(source_model, device, src_train_loader, source_optimizer, epoch)
            if (epoch+1) % 1 == 0:
                val_acc, _ = utils.utils.test(source_model, device, src_val_loader, split="val")
                out_str = '[Epoch: {}] Val Accuracy: {:.3f} '.format(epoch, val_acc)
                print(out_str) 
                
                if (val_acc > best_val_acc):
                    best_val_acc = val_acc
                    best_source_model = copy.deepcopy(source_model)
                     # チェックポイントを保存する前にディレクトリを作成
                    checkpoint_dir = os.path.join('checkpoints', 'source', args.dataset)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    # モデルを保存
                    torch.save(best_source_model.state_dict(), os.path.join(checkpoint_dir, source_file))

                    #torch.save(best_source_model.state_dict(), os.path.join(da_model_dir, f"round_0.pth"))
                    #utils.save_model_to_drive(service, best_source_model.state_dict(), da_model_dir, f"round_0.pth", DRIVE_ROOT_FOLDER_ID)
                    drive_folder = da_model_dir
                    drive_file = f"round_0.pth"
                    drive_utils.save_model_drive_relay_pc(best_source_model.state_dict(),drive_folder,drive_file)
                    
    best_source_model = torch.nn.DataParallel(best_source_model, device_ids=list(range(torch.cuda.device_count())))

    # Evaluate on source test set
    src_start_test_acc, _ = utils.utils.test(best_source_model, device, src_test_loader, split="test")
    print('{} Test Accuracy: {:.3f} '.format(args.source, src_start_test_acc))

    ############################################################################################
    # 各クラスの正解率を計算
    test_mode = "ori" 
    #per_class_accuracy = test_per_class_accuracy(best_model, device, target_test_loader, mode=test_mode, class_index_to_name=class_index_to_name)
    src_per_class_accuracy, src_per_class_accuracy_array = utils.utils.test_per_class_accuracy(best_source_model, device, src_test_loader, mode=test_mode)
    #print("Src Per-class Accuracy:", src_per_class_accuracy)

    # 結果をDataFrameに変換して保存
    src_per_class_accuracy_df = pd.DataFrame(list(src_per_class_accuracy.items()), columns=['Class', 'Accuracy (%)'])
    src_output_dir = os.path.join(exp_path,"src_per_class_accuracy")
    os.makedirs(src_output_dir,exist_ok=True)
    src_per_class_accuracy_csv_path = os.path.join(src_output_dir, f'src_per_class_accuracy.csv')
    src_per_class_accuracy_df.to_csv(src_per_class_accuracy_csv_path, index=False)
    print(f"Saved per-class accuracy to {src_per_class_accuracy_csv_path}")
    ##############################################################################################

    # Run active adaptation experiments
    target_accs = run_active_adaptation(args, best_source_model, src_dset, num_classes, device, writer, exp_path, src_start_test_acc, run_num)
    pp.pprint(target_accs)
    print("exp path: \t", exp_path)

if __name__ == '__main__':
    main()