J：保存済み 514 / 514 固有モデルを再評価。再学習なし。
条件所属は重複を許すが、全体・損失別の分布はモデルと設定の同一性で重複を除外する。Gの40中間モデルは追加しない。
平坦背景は各フレームの平均中心輝度 ≥ 0.65。Hのテクスチャ背景は平均（再構成中心 − 描画前背景中心）≥ 平均（物体輝度 − 描画前背景中心）/2。過去Hの判定は書き換えず、rawルールの比較列を残す。
集計のクラスは各モデルの5,000評価例における中心平均で決まり、個々の評価例の可視性を意味しない。
[モデル別](models.csv)・[条件所属別](memberships.csv)・[条件別4分類数](condition_counts.csv)・[プローブ分布](probe_distributions.csv)・[境界近傍](borderline.csv)・[Hの旧新判定](h_rule_comparison.csv)・[未学習25参照](untrained_reference.csv)。

条件別の数は、背景のみ / 過去のみ / 現在のみ / 両フレームの順。

| 群 | 役割 | r | ノイズ | 量 / 背景SD | 入れる場所 | 方法 | n / 予定 | 背景のみ | 過去のみ | 現在のみ | 両フレーム |
|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|
| A | primary | 1 | gaussian | 0 | target_only | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| A | primary | 1 | gaussian | 0 | target_only | l2 | 5 / 5 | 0 | 1 | 4 | 0 |
| A | primary | 1 | gaussian | 0.05 | target_only | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| A | primary | 1 | gaussian | 0.05 | target_only | l2 | 5 / 5 | 0 | 2 | 3 | 0 |
| A | primary | 1 | gaussian | 0.1 | target_only | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| A | primary | 1 | gaussian | 0.1 | target_only | l2 | 5 / 5 | 0 | 1 | 4 | 0 |
| A | primary | 1 | gaussian | 0.2 | target_only | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| A | primary | 1 | gaussian | 0.2 | target_only | l2 | 5 / 5 | 0 | 3 | 2 | 0 |
| A | primary | 1 | gaussian | 0.4 | target_only | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| A | primary | 1 | gaussian | 0.4 | target_only | l2 | 5 / 5 | 3 | 1 | 1 | 0 |
| A | primary | 2 | gaussian | 0 | target_only | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| A | primary | 2 | gaussian | 0 | target_only | l2 | 5 / 5 | 0 | 2 | 3 | 0 |
| A | primary | 2 | gaussian | 0.05 | target_only | l1 | 5 / 5 | 0 | 1 | 4 | 0 |
| A | primary | 2 | gaussian | 0.05 | target_only | l2 | 5 / 5 | 0 | 2 | 3 | 0 |
| A | primary | 2 | gaussian | 0.1 | target_only | l1 | 5 / 5 | 0 | 4 | 1 | 0 |
| A | primary | 2 | gaussian | 0.1 | target_only | l2 | 5 / 5 | 0 | 2 | 3 | 0 |
| A | primary | 2 | gaussian | 0.2 | target_only | l1 | 5 / 5 | 0 | 1 | 4 | 0 |
| A | primary | 2 | gaussian | 0.2 | target_only | l2 | 5 / 5 | 0 | 1 | 4 | 0 |
| A | primary | 2 | gaussian | 0.4 | target_only | l1 | 5 / 5 | 0 | 2 | 3 | 0 |
| A | primary | 2 | gaussian | 0.4 | target_only | l2 | 5 / 5 | 0 | 2 | 3 | 0 |
| A | primary | 3 | gaussian | 0 | target_only | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| A | primary | 3 | gaussian | 0 | target_only | l2 | 5 / 5 | 0 | 1 | 0 | 4 |
| A | primary | 3 | gaussian | 0.05 | target_only | l1 | 5 / 5 | 0 | 2 | 3 | 0 |
| A | primary | 3 | gaussian | 0.05 | target_only | l2 | 5 / 5 | 0 | 1 | 0 | 4 |
| A | primary | 3 | gaussian | 0.1 | target_only | l1 | 5 / 5 | 0 | 2 | 3 | 0 |
| A | primary | 3 | gaussian | 0.1 | target_only | l2 | 5 / 5 | 0 | 0 | 2 | 3 |
| A | primary | 3 | gaussian | 0.2 | target_only | l1 | 5 / 5 | 0 | 2 | 3 | 0 |
| A | primary | 3 | gaussian | 0.2 | target_only | l2 | 5 / 5 | 0 | 1 | 0 | 4 |
| A | primary | 3 | gaussian | 0.4 | target_only | l1 | 5 / 5 | 0 | 5 | 0 | 0 |
| A | primary | 3 | gaussian | 0.4 | target_only | l2 | 5 / 5 | 0 | 3 | 0 | 2 |
| A | primary | 5 | gaussian | 0 | target_only | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| A | primary | 5 | gaussian | 0 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 5 | gaussian | 0.05 | target_only | l1 | 5 / 5 | 0 | 2 | 3 | 0 |
| A | primary | 5 | gaussian | 0.05 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 5 | gaussian | 0.1 | target_only | l1 | 5 / 5 | 0 | 4 | 1 | 0 |
| A | primary | 5 | gaussian | 0.1 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 5 | gaussian | 0.2 | target_only | l1 | 5 / 5 | 0 | 1 | 1 | 3 |
| A | primary | 5 | gaussian | 0.2 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 5 | gaussian | 0.4 | target_only | l1 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 5 | gaussian | 0.4 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 8 | gaussian | 0 | target_only | l1 | 5 / 5 | 2 | 0 | 3 | 0 |
| A | primary | 8 | gaussian | 0 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 8 | gaussian | 0.05 | target_only | l1 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 8 | gaussian | 0.05 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 8 | gaussian | 0.1 | target_only | l1 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 8 | gaussian | 0.1 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 8 | gaussian | 0.2 | target_only | l1 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 8 | gaussian | 0.2 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 8 | gaussian | 0.4 | target_only | l1 | 5 / 5 | 0 | 0 | 0 | 5 |
| A | primary | 8 | gaussian | 0.4 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| B | primary | 1 | salt_pepper | 0.2 | target_only | l1 | 3 / 3 | 3 | 0 | 0 | 0 |
| B | primary | 1 | salt_pepper | 0.2 | target_only | l2 | 3 / 3 | 0 | 1 | 2 | 0 |
| B | primary | 1 | salt_pepper | 0.45 | target_only | l1 | 3 / 3 | 3 | 0 | 0 | 0 |
| B | primary | 1 | salt_pepper | 0.45 | target_only | l2 | 3 / 3 | 3 | 0 | 0 | 0 |
| B | primary | 2 | salt_pepper | 0.2 | target_only | l1 | 3 / 3 | 3 | 0 | 0 | 0 |
| B | primary | 2 | salt_pepper | 0.2 | target_only | l2 | 3 / 3 | 0 | 2 | 1 | 0 |
| B | primary | 2 | salt_pepper | 0.45 | target_only | l1 | 3 / 3 | 3 | 0 | 0 | 0 |
| B | primary | 2 | salt_pepper | 0.45 | target_only | l2 | 3 / 3 | 0 | 1 | 2 | 0 |
| B | primary | 3 | salt_pepper | 0.2 | target_only | l1 | 3 / 3 | 3 | 0 | 0 | 0 |
| B | primary | 3 | salt_pepper | 0.2 | target_only | l2 | 3 / 3 | 0 | 0 | 0 | 3 |
| B | primary | 3 | salt_pepper | 0.45 | target_only | l1 | 3 / 3 | 3 | 0 | 0 | 0 |
| B | primary | 3 | salt_pepper | 0.45 | target_only | l2 | 3 / 3 | 0 | 0 | 0 | 3 |
| B | primary | 5 | salt_pepper | 0.2 | target_only | l1 | 3 / 3 | 3 | 0 | 0 | 0 |
| B | primary | 5 | salt_pepper | 0.2 | target_only | l2 | 3 / 3 | 0 | 0 | 0 | 3 |
| B | primary | 5 | salt_pepper | 0.45 | target_only | l1 | 3 / 3 | 3 | 0 | 0 | 0 |
| B | primary | 5 | salt_pepper | 0.45 | target_only | l2 | 3 / 3 | 0 | 0 | 0 | 3 |
| B | primary | 8 | salt_pepper | 0.2 | target_only | l1 | 3 / 3 | 2 | 0 | 1 | 0 |
| B | primary | 8 | salt_pepper | 0.2 | target_only | l2 | 3 / 3 | 0 | 0 | 0 | 3 |
| B | primary | 8 | salt_pepper | 0.45 | target_only | l1 | 3 / 3 | 1 | 1 | 1 | 0 |
| B | primary | 8 | salt_pepper | 0.45 | target_only | l2 | 3 / 3 | 0 | 0 | 0 | 3 |
| D | primary | 5 | gaussian | 0.4 | both | l1 | 5 / 5 | 0 | 0 | 0 | 5 |
| D | primary | 5 | gaussian | 0.4 | both | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| D | primary | 5 | gaussian | 0.4 | input_only | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| D | primary | 5 | gaussian | 0.4 | input_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| D | primary | 5 | salt_pepper | 0.2 | both | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| D | primary | 5 | salt_pepper | 0.2 | both | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| D | primary | 5 | salt_pepper | 0.2 | input_only | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| D | primary | 5 | salt_pepper | 0.2 | input_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| D | target_only_control | 5 | gaussian | 0.4 | target_only | l1 | 5 / 5 | 0 | 0 | 0 | 5 |
| D | target_only_control | 5 | gaussian | 0.4 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| D | target_only_control | 5 | salt_pepper | 0.2 | target_only | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| D | target_only_control | 5 | salt_pepper | 0.2 | target_only | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 5 | gaussian | 0 | target_only | l1_6000 | 5 / 5 | 5 | 0 | 0 | 0 |
| G | primary | 5 | gaussian | 0 | target_only | l2_3000_l1_3000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 5 | gaussian | 0 | target_only | l2_6000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 5 | salt_pepper | 0.1 | target_only | l1_6000 | 5 / 5 | 5 | 0 | 0 | 0 |
| G | primary | 5 | salt_pepper | 0.1 | target_only | l2_3000_l1_3000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 5 | salt_pepper | 0.1 | target_only | l2_6000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 5 | salt_pepper | 0.2 | target_only | l1_6000 | 5 / 5 | 5 | 0 | 0 | 0 |
| G | primary | 5 | salt_pepper | 0.2 | target_only | l2_3000_l1_3000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 5 | salt_pepper | 0.2 | target_only | l2_6000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 5 | salt_pepper | 0.3 | target_only | l1_6000 | 5 / 5 | 5 | 0 | 0 | 0 |
| G | primary | 5 | salt_pepper | 0.3 | target_only | l2_3000_l1_3000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 5 | salt_pepper | 0.3 | target_only | l2_6000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 5 | salt_pepper | 0.45 | target_only | l1_6000 | 5 / 5 | 5 | 0 | 0 | 0 |
| G | primary | 5 | salt_pepper | 0.45 | target_only | l2_3000_l1_3000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 5 | salt_pepper | 0.45 | target_only | l2_6000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 8 | gaussian | 0 | target_only | l1_6000 | 5 / 5 | 2 | 0 | 3 | 0 |
| G | primary | 8 | gaussian | 0 | target_only | l2_3000_l1_3000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 8 | gaussian | 0 | target_only | l2_6000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 8 | salt_pepper | 0.2 | target_only | l1_6000 | 5 / 5 | 2 | 1 | 2 | 0 |
| G | primary | 8 | salt_pepper | 0.2 | target_only | l2_3000_l1_3000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 8 | salt_pepper | 0.2 | target_only | l2_6000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 8 | salt_pepper | 0.45 | target_only | l1_6000 | 5 / 5 | 1 | 2 | 2 | 0 |
| G | primary | 8 | salt_pepper | 0.45 | target_only | l2_3000_l1_3000 | 5 / 5 | 0 | 0 | 0 | 5 |
| G | primary | 8 | salt_pepper | 0.45 | target_only | l2_6000 | 5 / 5 | 0 | 0 | 0 | 5 |
| H | flat_control | 5 | なし（平坦背景） | 背景SD 0 | 入力・教師で共通 | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| H | flat_control | 5 | なし（平坦背景） | 背景SD 0 | 入力・教師で共通 | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| H | flat_control | 8 | なし（平坦背景） | 背景SD 0 | 入力・教師で共通 | l1 | 5 / 5 | 2 | 0 | 3 | 0 |
| H | flat_control | 8 | なし（平坦背景） | 背景SD 0 | 入力・教師で共通 | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| H | primary | 5 | なし（背景模様） | 背景SD 0.1 | 入力・教師で共通 | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| H | primary | 5 | なし（背景模様） | 背景SD 0.1 | 入力・教師で共通 | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| H | primary | 5 | なし（背景模様） | 背景SD 0.4 | 入力・教師で共通 | l1 | 5 / 5 | 5 | 0 | 0 | 0 |
| H | primary | 5 | なし（背景模様） | 背景SD 0.4 | 入力・教師で共通 | l2 | 5 / 5 | 5 | 0 | 0 | 0 |
| H | primary | 8 | なし（背景模様） | 背景SD 0.1 | 入力・教師で共通 | l1 | 5 / 5 | 0 | 0 | 2 | 3 |
| H | primary | 8 | なし（背景模様） | 背景SD 0.1 | 入力・教師で共通 | l2 | 5 / 5 | 0 | 0 | 0 | 5 |
| H | primary | 8 | なし（背景模様） | 背景SD 0.4 | 入力・教師で共通 | l1 | 5 / 5 | 0 | 0 | 0 | 5 |
| H | primary | 8 | なし（背景模様） | 背景SD 0.4 | 入力・教師で共通 | l2 | 5 / 5 | 1 | 0 | 2 | 2 |

プローブ値は中央値 [最小, 最大]。未学習参照は半径別5シードで、損失・再構成分類は適用しない。

| 母集団 | 損失 | 再構成分類 | r | n | 位置MLP R² | 速度MLP R² | 未来MLP R² |
|---|---|---|---|---:|---:|---:|---:|
| trained_unique | all | 背景のみ | all | 138 | 0.92677 [-0.08007, 0.99624] | 0.06647 [-0.08893, 0.85129] | 0.66046 [-0.10571, 0.92803] |
| trained_unique | all | 過去のみ | all | 57 | 0.81995 [0.74471, 0.87953] | 0.05107 [0.03545, 0.33737] | 0.42197 [0.25513, 0.62148] |
| trained_unique | all | 現在のみ | all | 79 | 0.99275 [0.31879, 0.99867] | 0.06360 [-0.06343, 0.10388] | 0.79480 [0.13755, 0.80790] |
| trained_unique | all | 両フレーム | all | 240 | 0.98923 [0.23285, 0.99742] | 0.96408 [-0.06891, 0.99219] | 0.97364 [0.09297, 0.99292] |
| trained_unique | l1 | 背景のみ | all | 126 | 0.92677 [-0.07852, 0.99624] | 0.08562 [-0.08893, 0.85129] | 0.66565 [-0.10571, 0.92803] |
| trained_unique | l1 | 過去のみ | all | 30 | 0.81530 [0.74471, 0.82905] | 0.05212 [0.04571, 0.08379] | 0.40018 [0.25513, 0.43054] |
| trained_unique | l1 | 現在のみ | all | 40 | 0.99383 [0.98849, 0.99867] | 0.06921 [0.05590, 0.10388] | 0.78529 [0.71455, 0.79788] |
| trained_unique | l1 | 両フレーム | all | 41 | 0.99346 [0.37517, 0.99705] | 0.97892 [-0.05885, 0.98964] | 0.97918 [0.17883, 0.99113] |
| trained_unique | l2 | 背景のみ | all | 12 | 0.62218 [-0.08007, 0.96936] | -0.03954 [-0.07834, 0.08443] | 0.38437 [-0.10193, 0.72432] |
| trained_unique | l2 | 過去のみ | all | 27 | 0.82941 [0.81648, 0.87953] | 0.04813 [0.03545, 0.33737] | 0.43030 [0.40144, 0.62148] |
| trained_unique | l2 | 現在のみ | all | 39 | 0.99149 [0.31879, 0.99538] | 0.05929 [-0.06343, 0.07198] | 0.79694 [0.13755, 0.80790] |
| trained_unique | l2 | 両フレーム | all | 159 | 0.98786 [0.23285, 0.99725] | 0.95978 [-0.06891, 0.99219] | 0.97271 [0.09297, 0.99292] |
| trained_unique | l2_to_l1 | 背景のみ | all | 0 | 該当なし | 該当なし | 該当なし |
| trained_unique | l2_to_l1 | 過去のみ | all | 0 | 該当なし | 該当なし | 該当なし |
| trained_unique | l2_to_l1 | 現在のみ | all | 0 | 該当なし | 該当なし | 該当なし |
| trained_unique | l2_to_l1 | 両フレーム | all | 40 | 0.99069 [0.98642, 0.99742] | 0.96884 [0.93728, 0.99183] | 0.97732 [0.96466, 0.99244] |
| untrained_reference | not_applicable | 適用なし | 1 | 5 | 0.77091 [0.70476, 0.80383] | -0.03365 [-0.03699, -0.02958] | 0.50738 [0.49516, 0.57605] |
| untrained_reference | not_applicable | 適用なし | 2 | 5 | 0.79987 [0.71694, 0.85196] | -0.00138 [-0.01746, 0.02195] | 0.52785 [0.48883, 0.64364] |
| untrained_reference | not_applicable | 適用なし | 3 | 5 | 0.83126 [0.74400, 0.88205] | 0.07688 [0.05027, 0.09421] | 0.56783 [0.51782, 0.67869] |
| untrained_reference | not_applicable | 適用なし | 5 | 5 | 0.88105 [0.80463, 0.94178] | 0.29763 [0.17628, 0.34463] | 0.67413 [0.58630, 0.74292] |
| untrained_reference | not_applicable | 適用なし | 8 | 5 | 0.92713 [0.81494, 0.96595] | 0.51573 [0.46723, 0.61995] | 0.77386 [0.64257, 0.80580] |

符号付き判定余裕の絶対値が0.05以下のモデル：9 件。境界近傍CSVに両フレームの値と元設定IDを保存。
Hの旧raw中心基準から分類が変わったモデル：0 / 60 件（平坦対照を含む）。
Aヒートマップは最頻クラスとその数を示す。同数首位は全該当クラスを明示し斜線表示する。

![Aの最頻再構成クラス](a_modal_classes.png)

境界近傍モデルの全件。余裕は中心基準の左辺 − 右辺（平坦背景は中心輝度 − 0.65）。

| 群 | r | ノイズ・量 | 背景SD | 入れる場所 | 方法 | seed | 過去中心 | 現在中心 | 過去余裕 | 現在余裕 | 分類 | 元ID |
|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| A | 1 | gaussian 0.4 | 0 | target_only | l2 | 4 | 0.65338 | 0.30489 | 0.00338 | -0.34511 | 過去のみ | e17911cd4dda |
| B | 1 | salt_pepper 0.2 | 0 | target_only | l2 | 0 | 0.68099 | 0.34393 | 0.03099 | -0.30607 | 過去のみ | 4d4c36c28468 |
| B | 1 | salt_pepper 0.2 | 0 | target_only | l2 | 1 | 0.34310 | 0.66886 | -0.30690 | 0.01886 | 現在のみ | 6315d2a17133 |
| B | 1 | salt_pepper 0.2 | 0 | target_only | l2 | 2 | 0.34140 | 0.66038 | -0.30860 | 0.01038 | 現在のみ | c2d7954345b4 |
| H | 8 | なし（背景模様） | 0.4 | 入力・教師で共通 | l2 | 0 | 0.64143 | 0.64405 | -0.00747 | -0.00223 | 背景のみ | dc8b2a4b8534 |
| H | 8 | なし（背景模様） | 0.4 | 入力・教師で共通 | l2 | 1 | 0.68872 | 0.68069 | 0.03982 | 0.03441 | 両フレーム | 356c5ef82401 |
| H | 8 | なし（背景模様） | 0.4 | 入力・教師で共通 | l2 | 2 | 0.66033 | 0.65499 | 0.01143 | 0.00871 | 両フレーム | 989bcb648287 |
| H | 8 | なし（背景模様） | 0.4 | 入力・教師で共通 | l2 | 3 | 0.64344 | 0.65125 | -0.00545 | 0.00497 | 現在のみ | 1c044da92468 |
| H | 8 | なし（背景模様） | 0.4 | 入力・教師で共通 | l2 | 4 | 0.64749 | 0.65527 | -0.00140 | 0.00898 | 現在のみ | 802da825127b |
