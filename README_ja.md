# drbench

[English](README.md)

次元削減手法を、同じデータと同じ指標で比べるためのベンチマーク基盤。
新しい手法を作る前に「何をもって良いとするか」を固定するために作った。
MADR の後継である [ScaleRank](https://github.com/tsjshg/scalerank-web) は、これを使って開発している。

## 使い方

```bash
uv sync
uv run python scripts/convert_legacy.py   # ../*.pkl（2020年の scRNA-seq）→ data/raw/*.npz
uv run drbench list
uv run drbench run                        # 既定: 8データ × 8手法 × seed 0, 1, 2（キャッシュあり）
uv run drbench run -d digits -m tsne:perplexity=5 umap:n_neighbors=50
uv run drbench report                     # results/report/ に図と summary.md
uv run pytest
```

埋め込みは `results/embeddings/` にキャッシュされ、足りない組み合わせだけ計算される。
指標を変えたら `--rescore`、手法を変えたら `--force`。

`data/` と `results/` はチェックアウトの中に置かれる。
別の場所に置くとき（パッケージとしてインストールしたときなど）は `DRBENCH_HOME` で指定する。

## 構成

| ファイル | 役割 |
|---|---|
| `src/drbench/metrics.py` | 評価指標。co-ranking 系は N×N 行列を作らず行チャンクで集計 |
| `src/drbench/datasets.py` | データセット登録。`X` は「手法の入力」かつ「指標の基準空間」 |
| `src/drbench/methods.py` | 手法登録。`@register("name")` で `fn(X, seed, **params) -> (N,2)` を足すだけ |
| `src/drbench/runner.py` | グリッド実行・キャッシュ・集計 |
| `src/drbench/report.py` | R_NX(K) 曲線、散布図、局所×大域のトレードオフ図、summary.md |
| `scripts/stability.py` | seed と行の順番で地図がどれだけ変わるか |

手法は `name` または `name:key=val,key=val`（例: `tsne:perplexity=5`）という文字列で指定するので、パラメータを変えた版にコードは要らない。

### ほかのパッケージの手法

drbench を変えずに、ほかのパッケージから手法を足せる。
そのパッケージの `pyproject.toml` で、`drbench.methods` グループのエントリポイントを宣言する。

```toml
[project.entry-points."drbench.methods"]
mymethods = "mypackage.drbench_methods"
```

drbench は起動時にそのモジュールを読み込むので、中の `@register` で手法が登録される。
パッケージを drbench の環境に入れれば、`drbench list` に出てくる。
ScaleRank の研究用コード（[`scalerank-web/research`](https://github.com/tsjshg/scalerank-web/tree/main/research)）はこの仕組みを使っている。

## 指標

| 指標 | 見ているもの | 備考 |
|---|---|---|
| `rnx@K` | スケール K での近傍保存（ランダム=0、完全=1） | Lee & Verleysen 2009 |
| `auc_rnx` | 全スケールの R_NX を 1/K で重み付けした面積 | Lee et al. 2015。局所寄り |
| `trust@K` / `cont@K` | 偽の近傍（intrusion）/ 失われた近傍（extrusion） | sklearn の trustworthiness と一致を確認済み |
| `dist_spearman` | 全点対距離の順位相関 | 大域構造 |
| `density_corr` | kNN 半径（局所密度）の対数の順位相関 | densMAP 論文の評価法 |
| `knn_acc` / `knn_acc_coarse` | ラベル / 上位クラスタの 10-NN 分類精度 | 参照値として元空間 X での精度も保存 |
| `footrule` | 全点対の \|順位差\| の平均 /(N-1)。ランダム≈1/3 | 2022年の「Qスコア」を正規化したもの |
| `qscore_legacy` | Σ\|順位差\| / N | 2022年の数値と直接比較できる |
| `overlap@K`（点ごと） | 各点の kNN 一致率 | 点ごとの信頼度マップの素材 |

1 万点を超えるデータでは、順位系の指標を 2000 行（アンカー）だけで推定する。
どの指標も行ごとの値の平均なので、推定に偏りはない。アンカーの数は `n_anchors` に記録する。
点ごとの `overlap@K` は、その場合も kNN リストから全点について計算する。

## データセット

- 合成: `s_curve`, `swiss_roll`（多様体）、`hier_blobs`（3×4 の階層クラスタ = 大域配置）、`density_blobs`（広がりの違う 4 クラスタ = 密度）
- 実データ: `digits`, `pollen`, `usoskin`, `kolodziejczyk`, `buettner`（scRNA-seq は CPM→log1p→分散上位2000遺伝子→PCA50）
- `mnist10k`, `fmnist10k`（ランダムな 1 万点）と `mnist70k`, `fmnist70k`（全 7 万点）、いずれも PCA50。初回に OpenML からダウンロード（ネットワーク必要）
