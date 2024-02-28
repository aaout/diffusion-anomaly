# Scripts

実験で使用したプログラムをデータ作成, 学習/生成, 実験に分けて説明する.

プログラムの優先度を ★ 三段階で評価している.

## データ作成

### create_dataset.py

    ★★☆(前処理したデータはRAIDに保存済み)
    各被験者の各チャネルのボクセルデータに対し、上位1%を1, 下位1%を0, その他を線形に正規化する関数

    Args:
        voxel_data (numpy.ndarray): 各被験者の各チャネルのボクセルデータ[240, 240, 155]

    Returns:
        normalized_data (numpy.ndarray): 0-1に正規化した後のボクセルデータ[240, 240, 155]

### create_clip_json.py

    ★★☆(前処理したデータはRAIDに保存済み)
    各被験者の各チャネルのボクセルデータに対し、clip点(上位1%と下位1%)の値を返す関数
    jsonファイルは/data/clip_train.json, /data/clip_test.jsonに保存
    作成したjsonファイルはscripts/create_dataset.pyで使用

    Args:
        voxel_data (numpy.ndarray [240, 240, 155]): 各被験者の各チャネルのボクセルデータ

    Returns:
        upper_clip (int): 上位1%の値
        lower_clip (int): 下位1%の値

## 学習 or 生成

### classifier_train.py

### classifier_valid.py

### classifier_sample_known.py

## 実験

### calc_score_and_save_binarized_data.py

    ★★★
    生成結果を二値化し, 二値化した結果とラベルデータからスコアを算出
    スライスごとの出力されるデータを被験者事に結合し, ボクセルごとにスコアの算出とデータの保存を行う

    Args:
        diffmap_dirs (str): 生成結果のディレクトリパス
        save_dir (str): 二値化したデータを保存するディレクトリパス

### abnormal_area_histgram.py

### calc_voxel_histgram.py

    ★☆☆(どれくらい差分があるかを確認するためのスクリプトなので確認の優先度は低い)
    4チャネルの入出力差分データのヒストグラムを表示

### TODO

-   [ ] Scripts の各ファイルについて説明
-   [ ] どのファイルを実行すればどのような処理がされるのか記載
