"""Is the map a function of the data, or of the random seed and the row order?

Two experiments on the 8 benchmark datasets, each comparing pairs of
embeddings of the same points:

* seed: the cached runs with seeds 0, 1, 2 (all three pairs);
* order: seed 0 on the rows in their original order and in 3 random
  permutations (the results are put back in the original order first).

For each pair it reports
  - procrustes: scipy's Procrustes disparity after the best shift, scaling,
    rotation and reflection (0 = same shape, 1 = unrelated);
  - knn: mean share of each point's 10 nearest neighbours that the two maps
    agree on (local stability);
  - dist: Spearman correlation of pairwise distances in the two maps, on
    random pairs (global stability).

    uv run python scripts/stability.py              # both experiments
    uv run python scripts/stability.py --skip-order # seed only, from the cache (keeps earlier order rows)
    uv run python scripts/stability.py --seed-methods tsne "mymethod:alpha=1" --order-methods tsne mymethod

The seed experiment reads seeds 0-2 from the cache (``drbench run`` them
first); methods missing a seed are skipped.

Writes results/stability.csv (one row per dataset, method, experiment and pair).

It also asks whether the points that move between seeds are the points the
map gets wrong: per point, the seed-to-seed neighbour agreement against the
share of its 10 nearest neighbours in the data that the map keeps (mean over
seeds), as a Spearman correlation per dataset (results/stability_points.csv).
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from drbench import datasets, methods
from drbench.runner import RESULTS_DIR, _paths

ORDER = ["s_curve", "hier_blobs", "density_blobs", "digits", "pollen", "usoskin", "kolodziejczyk", "buettner"]
SEED_METHODS = [
    "pca", "mds", "tsne", "tsne:initialization=random",
    "umap", "umap:init=pca", "umap:init=random", "densmap", "pacmap", "trimap",
]
ORDER_METHODS = ["pca", "tsne", "umap", "umap:init=pca"]
POINT_METHODS = ["tsne", "umap:init=pca"]
K = 10


def knn_sets(Y):
    return NearestNeighbors(n_neighbors=K + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]


def compare(Y1, Y2, rng, n_pairs=20_000):
    _, _, disparity = procrustes(Y1, Y2)
    a, b = knn_sets(Y1), knn_sets(Y2)
    knn = np.mean([len(np.intersect1d(x, y)) / K for x, y in zip(a, b)])
    n = len(Y1)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = i != j
    d1 = np.linalg.norm(Y1[i[keep]] - Y1[j[keep]], axis=1)
    d2 = np.linalg.norm(Y2[i[keep]] - Y2[j[keep]], axis=1)
    return {"procrustes": disparity, "knn": knn, "dist": spearmanr(d1, d2).statistic}


def seed_rows(rng, specs):
    rows = []
    for ds in ORDER:
        for spec in specs:
            paths = [_paths(ds, spec, s)["emb"] for s in (0, 1, 2)]
            if not all(p.exists() for p in paths):
                print(f"  {ds:14s} {spec:28s} missing seeds, skipped")
                continue
            Ys = [np.load(p) for p in paths]
            for s, t in itertools.combinations(range(3), 2):
                rows.append({"dataset": ds, "method": spec, "experiment": "seed", "pair": f"{s}-{t}", **compare(Ys[s], Ys[t], rng)})
    return rows


def order_rows(rng, specs, n_perm=3):
    rows = []
    for ds in ORDER:
        X = datasets.load(ds).X
        perms = [np.random.default_rng(100 + r).permutation(len(X)) for r in range(n_perm)]
        for spec in specs:
            base = methods.run(spec, X, seed=0)
            for r, perm in enumerate(perms):
                Yp = np.empty_like(base)
                Yp[perm] = methods.run(spec, X[perm], seed=0)  # row k of the output belongs to point perm[k]
                rows.append({"dataset": ds, "method": spec, "experiment": "order", "pair": f"p{r}", **compare(base, Yp, rng)})
            print(f"  {ds:14s} {spec:28s} done", flush=True)
    return rows


def overlap(a, b):
    return np.array([len(np.intersect1d(x, y)) / K for x, y in zip(a, b)])


def point_rows(specs):
    rows = []
    for ds in ORDER:
        kx = knn_sets(datasets.load(ds).X)
        for spec in specs:
            paths = [_paths(ds, spec, s)["emb"] for s in (0, 1, 2)]
            if not all(p.exists() for p in paths):
                continue
            ks = [knn_sets(np.load(p)) for p in paths]
            stab = np.mean([overlap(ks[s], ks[t]) for s, t in itertools.combinations(range(3), 2)], axis=0)
            kept = np.mean([overlap(kx, k) for k in ks], axis=0)
            rows.append({"dataset": ds, "method": spec, "rho": spearmanr(stab, kept).statistic, "stability": stab.mean(), "kept": kept.mean()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-order", action="store_true", help="seed experiment only (keeps earlier order rows in the csv)")
    ap.add_argument("--seed-methods", nargs="+", default=SEED_METHODS, metavar="SPEC")
    ap.add_argument("--order-methods", nargs="+", default=ORDER_METHODS, metavar="SPEC")
    ap.add_argument("--point-methods", nargs="+", default=POINT_METHODS, metavar="SPEC")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = seed_rows(rng, args.seed_methods)
    out = RESULTS_DIR / "stability.csv"
    if not args.skip_order:
        rows += order_rows(rng, args.order_methods)
    elif out.exists():
        old = pd.read_csv(out)
        rows += old[old.experiment == "order"].to_dict("records")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    for exp in df.experiment.unique():
        t = df[df.experiment == exp].groupby("method", sort=False)[["procrustes", "knn", "dist"]].mean()
        print(f"\n== {exp}: mean over datasets and pairs")
        print(t.round(3).to_string())
    print(f"\nwritten {out}")
    pts = pd.DataFrame(point_rows(args.point_methods))
    pts.to_csv(RESULTS_DIR / "stability_points.csv", index=False)
    print("\n== per point: Spearman(seed-to-seed agreement, neighbours kept), mean over datasets")
    print(pts.groupby("method", sort=False)[["rho", "stability", "kept"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
