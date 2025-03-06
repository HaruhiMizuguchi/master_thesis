import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F

def distance_and_ui_weighted(ui_indices, cc_indices, unlab_component_conf, unlabeled_tgt_extracted_feature, unlabeled_tgt_preds, \
    n, labeled_target_loader, unlabeled_tgt_logits, unlabeled_tgt_labels,unlab_UC_conf,weights):
    # 重みの展開
    alpha = weights

    # 特徴ベクトル、ロジット、予測確率を UI に絞り込む
    ui_features = unlabeled_tgt_extracted_feature[ui_indices].cpu().numpy()
    ui_logits = unlabeled_tgt_logits[ui_indices].cpu().numpy()
    ui_probs = F.softmax(unlabeled_tgt_logits[ui_indices], dim=1).cpu().numpy()
    ui_preds = unlabeled_tgt_preds[ui_indices].numpy()
    ui_GMMprobs = unlab_component_conf
    
    # 1. CCカテゴリの予測結果からマイノリティクラスを特定
    cc_preds = unlabeled_tgt_preds[cc_indices].numpy().astype(int)  # 浮動小数点型を整数型に変換
    cc_class_counts = np.bincount(cc_preds, minlength=50)  # クラス数は50と仮定
    num_minority = len(cc_class_counts) // 2  # 少数クラスをクラスの半分と定義
    minority_classes = np.argsort(cc_class_counts)[:num_minority]  # 出現頻度が少ないクラスを取得


    # 2. マイノリティクラスごとの中心を計算
    cc_features = unlabeled_tgt_extracted_feature[cc_indices].cpu().numpy()
    class_centers = {}
    for cls in minority_classes:
        cls_indices = np.where(cc_preds == cls)[0]
        if len(cls_indices) > 0:
            class_centers[cls] = np.mean(cc_features[cls_indices], axis=0)
        else:
            # CC 内にサンプルが存在しない場合、全カテゴリでそのクラスと予測されたサンプルを使用
            all_cls_indices = np.where(unlabeled_tgt_preds.cpu().numpy() == cls)[0]
            if len(all_cls_indices) > 0:
                class_centers[cls] = np.mean(unlabeled_tgt_extracted_feature.cpu().numpy()[all_cls_indices], axis=0)
            else:
                # それでもサンプルがない場合は、クラス中心をゼロベクトルに設定
                class_centers[cls] = np.zeros(cc_features.shape[1])
                print(f"クラス {cls} のサンプルが存在しません。")

    # 3. UI の中でマジョリティクラスと予測されたデータをフィルタリング
    majority_classes = np.argsort(cc_class_counts)[num_minority:]  # 出現頻度が多いクラスを取得
    #majority_ui_indices = [idx for idx in range(len(ui_indices)) if ui_preds[idx] in majority_classes]
    majority_ui_indices = [idx for idx in range(len(ui_indices))]
    majority_ui_features = ui_features[majority_ui_indices]
    majority_ui_logits = ui_logits[majority_ui_indices]
    majority_ui_probs = ui_probs[majority_ui_indices]
    majority_ui_preds = ui_preds[majority_ui_indices]
    majority_ui_GMMprobs = ui_GMMprobs[majority_ui_indices]

    # 4. スコア計算
    # 4.1 距離スコア: マジョリティクラス内のサンプルとマイノリティクラス中心との距離
    distance_scores_dict = {}
    for cls in class_centers:
        center = class_centers[cls]
        distances = pairwise_distances(majority_ui_features, [center]).flatten()
        scaler = MinMaxScaler()
        if len(distances) > 1:  # サンプルが複数ある場合に正規化
            distances_normalized = scaler.fit_transform(distances.reshape(-1, 1)).flatten()
        else:
            distances_normalized = np.zeros_like(distances)  # サンプルが1つの場合はスコア0
        # 距離スコアを反転（近いほど高スコアにする）
        distance_scores_dict[cls] = 1 - distances_normalized


    # 5. 各少数クラスに対して1つずつサンプルを選択
    selected_idxs = []
    available_indices = set(range(len(majority_ui_indices)))  # 選択可能なインデックスの集合

    for cls in class_centers:
        ui_class_indices = [i for i in range(len(majority_ui_indices)) if majority_ui_preds[i] == cls]
        if len(ui_class_indices) != 0:
            # 各クラスのスコアを取得
            distances = distance_scores_dict[cls][ui_class_indices]
            ui_probs = majority_ui_GMMprobs[ui_class_indices]
            # MinMaxScalerでスコアを正規化（距離スコアとUIスコアを同時に処理）
            scaler = MinMaxScaler()
            if len(ui_class_indices) > 1:  # サンプルが複数ある場合に正規化
                distances_normalized = scaler.fit_transform(distances.reshape(-1, 1)).flatten()
                ui_probs_normalized = scaler.fit_transform(ui_probs.reshape(-1, 1)).flatten()
            else:
                # サンプルが1つの場合はスコアをそのまま使用
                distances_normalized = np.zeros_like(distances)
                ui_probs_normalized = np.zeros_like(ui_probs)
            # uiの確率とCC中心と距離を足したスコア
            combined_scores = (1-alpha)*ui_probs_normalized + alpha * distances_normalized
            # 総合スコアが最大のサンプルを取得
            max_index = ui_class_indices[np.argmax(combined_scores)]
            if max_index in available_indices:
                sample_index = ui_indices[majority_ui_indices[max_index]]
            selected_idxs.append(sample_index)
            # 使用済みインデックスを更新
            available_indices.remove(max_index)
        else:
            sample_index = np.argmax(ui_GMMprobs)
            selected_idxs.append(sample_index)

    return np.array(selected_idxs)

def margin(ui_indices, cc_indices, unlab_component_conf, unlabeled_tgt_extracted_feature, unlabeled_tgt_preds, \
    n, labeled_target_loader, unlabeled_tgt_logits, unlabeled_tgt_labels,unlab_UC_conf):
    # ソフトマックスを適用して確率に変換
    unlabeled_tgt_probs = torch.softmax(unlabeled_tgt_logits, dim=1)
    # 上位2つの確率を取得
    top2_probs, _ = torch.topk(unlabeled_tgt_probs, k=2, dim=1)
    # 最大値と2番目の値の差を計算
    unlabeled_uncertainty = top2_probs[:, 0] - top2_probs[:, 1]
    selected_idxs = np.argsort(unlabeled_uncertainty)[:n]
    return np.array(selected_idxs)
        
        
def entropy(ui_indices, cc_indices, unlab_component_conf, unlabeled_tgt_extracted_feature, unlabeled_tgt_preds, \
    n, labeled_target_loader, unlabeled_tgt_logits, unlabeled_tgt_labels,unlab_UC_conf):
    
    #ソフトマックスを適用して確率に変換
    unlabeled_tgt_probs = torch.softmax(unlabeled_tgt_logits, dim=1)
    # エントロピーを計算(符号を逆にし、0以下にする)
    unlabeled_entropy = -(-torch.sum(unlabeled_tgt_probs * torch.log(unlabeled_tgt_probs + 1e-10), dim=1))
    selected_idxs = np.argsort(unlabeled_entropy)[:n]
    return np.array(selected_idxs)
    
    
def random(ui_indices, cc_indices, unlab_component_conf, unlabeled_tgt_extracted_feature, unlabeled_tgt_preds, \
    n, labeled_target_loader, unlabeled_tgt_logits, unlabeled_tgt_labels,unlab_UC_conf):
    num_unlabeled = len(unlabeled_tgt_labels)
    selected_idxs = np.random.choice(num_unlabeled, n, replace=False)
    return selected_idxs