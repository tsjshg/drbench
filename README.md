# drbench

[日本語](README_ja.md)

A benchmark harness for comparing dimensionality-reduction methods on the
same datasets with the same metrics. It was built to fix "what counts as
better" before developing a new method:
[ScaleRank](https://github.com/tsjshg/scalerank-web), the successor of the
earlier MADR experiments, is developed against it.

## Usage

```bash
uv sync
uv run python scripts/convert_legacy.py   # ../*.pkl (the 2020 scRNA-seq data) → data/raw/*.npz
uv run drbench list
uv run drbench run                        # default: 8 datasets × 8 methods × seeds 0, 1, 2 (cached)
uv run drbench run -d digits -m tsne:perplexity=5 umap:n_neighbors=50
uv run drbench report                     # figures and summary.md in results/report/
uv run pytest
```

Embeddings are cached in `results/embeddings/`, and only missing
combinations are computed. After changing a metric use `--rescore`; after
changing a method, `--force`.

`data/` and `results/` live in the checkout. Set `DRBENCH_HOME` to keep them
somewhere else (for example when drbench is installed as a package).

## Layout

| File | Role |
|---|---|
| `src/drbench/metrics.py` | Quality metrics. Co-ranking metrics are accumulated over row chunks, without an N × N matrix |
| `src/drbench/datasets.py` | Dataset registry. `X` is both the method's input and the reference space for the metrics |
| `src/drbench/methods.py` | Method registry. A method is `fn(X, seed, **params) -> (N, 2)` decorated with `@register("name")` |
| `src/drbench/runner.py` | Grid runs, cache, aggregation |
| `src/drbench/report.py` | R_NX(K) curves, scatter plots, local × global trade-off plot, summary.md |
| `scripts/stability.py` | How much the maps change with the seed and with the row order |

Methods are addressed by spec strings, `name` or `name:key=val,key=val`
(e.g. `tsne:perplexity=5`), so parameter variants need no extra code.

### Methods from other packages

A package can add methods without changing drbench. Declare an entry point
in the `drbench.methods` group in its `pyproject.toml`:

```toml
[project.entry-points."drbench.methods"]
mymethods = "mypackage.drbench_methods"
```

drbench imports that module at start-up, so its `@register` calls add the
methods; install the package into drbench's environment and they show up in
`drbench list`. ScaleRank's research code
([`scalerank-web/research`](https://github.com/tsjshg/scalerank-web/tree/main/research))
works this way.

## Metrics

| Metric | What it measures | Notes |
|---|---|---|
| `rnx@K` | Neighbourhood preservation at scale K (random = 0, perfect = 1) | Lee & Verleysen 2009 |
| `auc_rnx` | Area under R_NX over all scales, weighted by 1/K | Lee et al. 2015; leans local |
| `trust@K` / `cont@K` | False neighbours (intrusions) / lost neighbours (extrusions) | Checked against sklearn's trustworthiness |
| `dist_spearman` | Rank correlation of all pairwise distances | Global structure |
| `density_corr` | Rank correlation of the log kNN radius (local density) | As used to evaluate densMAP |
| `knn_acc` / `knn_acc_coarse` | 10-NN classification accuracy of the labels / top-level clusters | The accuracy in the original space X is stored as a reference |
| `footrule` | Mean \|rank difference\| over all pairs / (N − 1); random ≈ 1/3 | The 2022 "Q score", normalised |
| `qscore_legacy` | Σ \|rank difference\| / N | Directly comparable with the 2022 numbers |
| `overlap@K` (per point) | Each point's kNN agreement | Material for a per-point reliability map |

Above 10,000 points the rank metrics are estimated from 2,000 random rows
(anchors). Every rank metric is a mean over rows, so the estimate is
unbiased; the number of anchors is stored as `n_anchors`. Per-point
`overlap@K` is then computed for all points from kNN lists.

## Datasets

- Synthetic: `s_curve`, `swiss_roll` (manifolds), `hier_blobs` (3 × 4 hierarchical clusters: global arrangement), `density_blobs` (4 clusters of different spread: density)
- Real: `digits`, `pollen`, `usoskin`, `kolodziejczyk`, `buettner` (scRNA-seq: CPM → log1p → top 2,000 variable genes → PCA-50)
- `mnist10k`, `fmnist10k` (10,000 random points) and `mnist70k`, `fmnist70k` (all 70,000), PCA-50: downloaded from OpenML on first use (needs a network connection)
