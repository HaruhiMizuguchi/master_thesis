import os

def split_data_by_classes(data_file, split_class_groups, new_class_labels, output_prefix, domain, mode):
    # 元のデータを読み込む
    with open(data_file, 'r') as f:
        data_lines = f.readlines()
    
    # 各分割グループごとに処理
    for i, class_group in enumerate(split_class_groups):
        new_labels = {cls: new_class_labels[cls] for cls in class_group}  # 新しいラベルを取得
        output_file = f'{output_prefix}/{domain}_{i + 1}_{mode}.txt'
        with open(output_file, 'w') as f:
            for line in data_lines:
                img_path, old_label = line.strip().split()
                old_label = int(old_label)
                
                # 対応するクラスに画像があるかを確認し、新しいラベルで書き出す
                class_name = img_path.split('/')[1]  # クラス名は画像パスの2番目の要素
                if class_name in new_labels:
                    new_label = new_labels[class_name]
                    f.write(f'{img_path} {new_label}\n')

def process_all_domains(base_dir, domains, split_class_groups, new_class_labels):
    output_prefix = os.path.join(base_dir, 'train_val_test_split')  # 出力ディレクトリのパスを指定
    os.makedirs(output_prefix, exist_ok=True)  # ディレクトリが存在しない場合は作成

    for domain in domains:
        for mode in ['train', 'test']:
            data_file = f'{base_dir}/{domain}_{mode}.txt'
            split_data_by_classes(data_file, split_class_groups, new_class_labels, output_prefix, domain, mode)

if __name__ == "__main__":
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    base_dir = 'train_val_test'

    # split_{i}.txt ファイルからクラス分割と新しいラベルを読み込む
    split_class_groups = []
    new_class_labels = {}
    for i in range(1, 8):  # 分割は6つの50クラス、1つの45クラス
        with open(f'split_{i}.txt', 'r') as f:
            class_group = []
            for line in f.readlines():
                cls, new_label = line.strip().split()
                class_group.append(cls)
                new_class_labels[cls] = int(new_label)
            split_class_groups.append(class_group)
    
    process_all_domains(base_dir, domains, split_class_groups, new_class_labels)
