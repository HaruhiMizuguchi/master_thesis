import random

def split_classes(class_to_label, split_sizes):
    # クラスをランダムにシャッフル
    classes = list(class_to_label.keys())
    random.shuffle(classes)
    
    split_class_groups = []
    start_idx = 0
    for size in split_sizes:
        split_class_groups.append(classes[start_idx:start_idx + size])
        start_idx += size
    
    return split_class_groups

def save_split_classes_with_new_labels(split_class_groups):
    # 各分割グループをテキストファイルに保存（アルファベット順にソートし、0から新しいラベルを付与）
    for i, class_group in enumerate(split_class_groups):
        sorted_class_group = sorted(class_group, key=lambda x: x.lower())  # 大文字小文字区別なしでソート
        with open(f'split_{i + 1}.txt', 'w') as f:
            for new_label, cls in enumerate(sorted_class_group):
                f.write(f'{cls} {new_label}\n')  # 新しいラベルを0から付与

if __name__ == "__main__":
    # class_labels.txt からクラスとラベルの対応を読み込む
    with open('class_labels.txt', 'r') as f:
        class_to_label = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}
    
    split_sizes = [50, 50, 50, 50, 50, 50, 45]  # 50クラスごとの分割サイズ
    
    # ランダムにクラスを分割
    split_class_groups = split_classes(class_to_label, split_sizes)
    
    # 新しいラベル番号を付けて分割結果を保存
    save_split_classes_with_new_labels(split_class_groups)
