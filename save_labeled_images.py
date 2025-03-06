# -*- coding: utf-8 -*-
import os
import shutil  # 追加

def get_class_from_path(image_path):
    """
    画像パスからクラス名を抽出します。

    Args:
        image_path (str): 画像のパス。

    Returns:
        str: クラス名。
    """
    parts = image_path.replace('\\','/').split('/')
    if len(parts) >= 2:
        class_name = parts[-2].lower().replace(' ', '_')
        return class_name
    else:
        return 'Unknown'

def read_class_labels(split_file):
    """
    split_fileからクラス番号とクラス名のマッピングを読み込みます。

    Args:
        split_file (str): splitファイルのパス。

    Returns:
        tuple: (class_index_to_name, class_name_to_index) の辞書。
    """
    class_index_to_name = {}
    class_name_to_index = {}
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                class_name = ' '.join(parts[:-1]).lower().replace(' ', '_')
                class_index = int(parts[-1])
                class_index_to_name[class_index] = class_name
                class_name_to_index[class_name] = class_index
    return class_index_to_name, class_name_to_index

def copy_labeled_images():
    # --- パラメータの設定 ---
    domain_idx = 0
    subset_idx = 1
    sampling_idx = 1
    for domain_idx in [0,1]:
        for subset_idx in [1,2]:
            for sampling_idx in [0,1,2]:
                samplings = ["ui", "per_round_class_balance","uncertainty_round_class_balance"]
                sampling = samplings[sampling_idx]
                source_domains = ["clipart", "real"]
                target_domains = ["sketch", "clipart"]
                source_domain = source_domains[domain_idx]
                target_domain = target_domains[domain_idx]

                run_nums = {
                    "sketch": {1: 1, 2: 1},
                    "clipart": {1: 1, 2: 1}
                }
                #run_num = run_nums[target_domain][subset_idx]
                run_num=1

                # データのパス設定/home/haruhi_mizuguchi/master_thesis/ADA/DiaNA/only_tgt_sup_exp_record_6r_150b_fix_seed_not_augment
                base_dir = os.path.join(
                    'only_tgt_sup_exp_record_6r_150b_fix_seed_not_augment', 'multi-round', "domainnet_50",
                    f'{source_domain}2{target_domain}',
                    'source_GMM_self_ft_6r_150b_ResNet34',
                    'warmupft-adapt_lr1e-05-wd1e-05-srcw0.1-ccw0.5-ucw0.1',
                    f"subset_{subset_idx}",
                    f"{sampling}",
                    f'{run_num}'
                )
                round_files_dir = os.path.join(base_dir, 'selected_image')
                data_output_dir = os.path.join(round_files_dir, 'list')
                os.makedirs(data_output_dir, exist_ok=True)
                # split_fileのパス
                split_file = f'../data/domainnet_50/split_{subset_idx}.txt'
                if not os.path.exists(split_file):
                    print(f"Split file not found: {split_file}")
                    return

                # クラスラベルの読み込み
                class_index_to_name, class_name_to_index = read_class_labels(split_file)

                # ラウンドごとの処理
                for round_num in range(1, 7):
                    round_file = os.path.join(round_files_dir, f'round_{round_num}.txt')
                    if not os.path.exists(round_file):
                        print(f"Round file not found: {round_file}")
                        continue
                    print(f"Processing Round {round_num}: {round_file}")

                    # ラウンドファイルから画像パスを読み込み
                    with open(round_file, 'r') as f:
                        image_paths = [line.strip() for line in f if line.strip()]

                    # 画像をクラスごとにコピー
                    for img_path in image_paths:
                        class_name = get_class_from_path(img_path)
                        if class_name == 'Unknown':
                            print(f"Warning: Could not determine class for image: {img_path}")
                            continue
                        if class_name in class_name_to_index:
                            dest_dir = os.path.join(data_output_dir, class_name, str(round_num))
                            os.makedirs(dest_dir, exist_ok=True)
                            dest_path = os.path.join(dest_dir, os.path.basename(img_path))
                            img_path = f"../data/domainnet/{img_path}"
                            if not os.path.exists(dest_path):
                                try:
                                    shutil.copy(img_path, dest_path)
                                except Exception as e:
                                    print(f"Error copying {img_path} to {dest_path}: {e}")
                        else:
                            print(f"Warning: Class '{class_name}' not found in class_name_to_index.")
                            continue

if __name__ == '__main__':
    copy_labeled_images()
