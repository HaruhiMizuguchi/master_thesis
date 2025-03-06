import random

def split_train_val_from_txt(domain,train_txt_path, val_rate=0.1):
  """
  {domain}_train.txt から train データと validation データを分割する

  Args:
    train_txt_path: {domain}_train.txt ファイルへのパス (e.g., 'clipart_train.txt')
    val_rate: validationデータの割合 (e.g., 0.1)
  """
  with open(train_txt_path, 'r') as f:
    data = [line.strip().split(' ') for line in f]

  random.shuffle(data)

  split_idx = int(len(data) * (1 - val_rate))
  train_data = data[:split_idx]
  val_data = data[split_idx:]

  #domain_name = train_txt_path.split('_')[0]  # ファイル名からドメイン名を取得
  domain_name = domain
  save_folder = "data/domainNet"
  write_data_to_txt(train_data, f"{save_folder}/{domain_name}_train.txt")
  write_data_to_txt(val_data, f"{save_folder}/{domain_name}_val.txt")

def write_data_to_txt(data, filename):
  """
  データのリストをtxtファイルに書き込む

  Args:
    data: 画像パスとラベルのペアのリスト
    filename: 出力ファイル名
  """
  with open(filename, 'w') as f:
    for image_path, label in data:
      f.write(f"{image_path} {label}\n")

if __name__ == '__main__':
    domains = ["clipart","infograph","painting","quickdraw","real","sketch"]
    for domain in domains:
        train_txt_path = f'data/domainNet/train_txt_original/{domain}_train.txt' # 適宜変更
        split_train_val_from_txt(domain,train_txt_path)