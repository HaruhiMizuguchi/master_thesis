def create_class_label_mapping_from_txt(txt_file):
    class_to_label = {}
    
    with open(txt_file, 'r') as f:
        for line in f:
            img_path, label = line.strip().split()
            class_name = img_path.split('/')[1]  # 2番目の要素がクラス名
            label = int(label)
            # クラス名が既に登録されていなければ追加
            if class_name not in class_to_label:
                class_to_label[class_name] = label
    
    # クラスとラベルの対応をテキストファイルに保存
    with open('class_labels.txt', 'w') as f:
        for cls, label in sorted(class_to_label.items(), key=lambda x: x[1]):  # ラベル順にソート
            f.write(f'{cls} {label}\n')
    
    return class_to_label

# 使用例
txt_file = 'train_val_test/clipart_train.txt'
class_label_mapping = create_class_label_mapping_from_txt(txt_file)
