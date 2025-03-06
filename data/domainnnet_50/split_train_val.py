import random
import os

def split_train_val_from_txt(domain, i, train_txt_path, val_rate=0.1):
    """
    {domain}_{i}_train.txt から train データと validation データを分割し、ソートする

    Args:
        domain: ドメイン名 (e.g., 'clipart')
        i: 分割の番号 (e.g., 1, 2, 3, ...)
        train_txt_path: {domain}_{i}_train.txt ファイルへのパス
        val_rate: validationデータの割合 (e.g., 0.1)
    """
    # データを読み込む
    with open(train_txt_path, 'r') as f:
        data = [line.strip().split(' ') for line in f]

    random.shuffle(data)

    # train と validation のデータを分割
    split_idx = int(len(data) * (1 - val_rate))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # 出力ディレクトリの作成
    save_folder = "train_test_split"
    os.makedirs(save_folder, exist_ok=True)

    # 分割された train と validation データを書き出し（ソート済み）
    write_data_to_txt(train_data, f"{save_folder}/{domain}_{i}_train.txt")
    write_data_to_txt(val_data, f"{save_folder}/{domain}_{i}_val.txt")

def write_data_to_txt(data, filename):
    """
    データのリストをtxtファイルに書き込む (画像パスでソート)

    Args:
        data: 画像パスとラベルのペアのリスト
        filename: 出力ファイル名
    """
    # 画像パスを基準にソート
    sorted_data = sorted(data, key=lambda x: x[0])

    with open(filename, 'w') as f:
        for image_path, label in sorted_data:
            f.write(f"{image_path} {label}\n")

if __name__ == '__main__':
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    base_dir = '../domainnet'

    # 各ドメインに対して50クラスに分割されたデータを処理
    for domain in domains:
        for i in range(1, 8):  # 1〜7の分割 (50クラスごと)
            train_txt_path = f'train_val_test_split/{domain}_{i}_train.txt'
            split_train_val_from_txt(domain, i, train_txt_path, val_rate=0.1)
