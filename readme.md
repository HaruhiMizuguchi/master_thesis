## データのダウンロードと準備

1. [DomainNet データセット](http://ai.bu.edu/M3SDA/)をダウンロードします。

2. ダウンロードしたデータを以下のような構造で配置します：
   ```
   data/
   └── domainnet/
       ├── clipart/
       │   ├── airplane/
       │   │   ├── 1.jpg
       │   │   └── ...
       │   └── ...
       ├── infograph/
       ├── painting/
       ├── quickdraw/
       ├── real/
       └── sketch/
   ```
## データのクラス部分集合への分割

DomainNetデータセットの345クラスを50クラスずつの部分集合に分割するには、以下の手順で実行してください：

1. クラスとラベルのマッピングを作成します：
   ```
   python data/domainnnet_50/get_class.py
   ```
   これにより`class_labels.txt`ファイルが生成されます。

2. クラスを50クラスずつのグループに分割します：
   ```
   python data/domainnnet_50/split_class.py
   ```
   これにより`split_1.txt`から`split_7.txt`までのファイルが生成されます（6つの50クラスグループと1つの45クラスグループ）。

3. 各ドメインのデータを分割したクラスグループごとに分けます：
   ```
   python data/domainnnet_50/split_data.py
   ```
   これにより各ドメインのトレーニングデータとテストデータが、クラスグループごとに分割されます。

4. トレーニングデータから検証データを分割します：
   ```
   python data/domainnnet_50/split_train_val.py
   ```
   これにより、各クラスグループのトレーニングデータの一部（デフォルトで10%）が検証データとして分割されます。

分割後のデータは以下のディレクトリ構造で保存されます：
```
data/domainnnet_50/train_val_test_split/
├── clipart_1_train.txt
├── clipart_1_val.txt
├── clipart_1_test.txt
├── clipart_2_train.txt
...
└── sketch_7_test.txt
```

各ファイルには、対応するドメインとクラスグループの画像パスとラベル（0から49まで）が含まれています。 

## モデルの訓練

モデルの訓練は以下のように実行します：

```
bash run_training.sh
```

このスクリプトには、ドメイン適応のための様々な設定が含まれています。必要に応じて`run_training.sh`を編集し、ソースドメイン、ターゲットドメイン、使用するクラスグループなどのパラメータを調整できます。

主な設定パラメータ：
- ソースドメイン（例：clipart）
- ターゲットドメイン（例：sketch）
- クラスグループ番号（1〜7）


