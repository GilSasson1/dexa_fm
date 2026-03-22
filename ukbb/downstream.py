import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_rel
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
except Exception:
    StratifiedGroupKFold = None


_VISIT_SUFFIX_RE = re.compile(r"\s*-\s*visit\s*(\d+)\s*$", re.IGNORECASE)

_SEED_LEVEL_COLUMNS = [
    "Seed",
    "Target",
    "Visit",
    "Source",
    "Task",
    "Metric",
    "N",
    "Score",
    "Score SD (folds)",
    "R2",
]

_SUMMARY_COLUMNS = [
    "Target",
    "Visit",
    "Source",
    "Task",
    "Metric",
    "Mean Score",
    "SE Score",
    "N Seeds",
    "N",
    "Mean R2",
]

_PVAL_COLUMNS = [
    "Target",
    "Visit",
    "Task",
    "Metric",
    "Baseline",
    "Source",
    "N Pairs",
    "T Statistic",
    "P Value",
]


def _default_seeds(n_seeds: int = 5, base: int = 73) -> List[int]:
    return [int(base * (10 ** i)) for i in range(max(1, n_seeds))]


def _clean_index_ids(index_like: pd.Index) -> pd.Index:
    idx = pd.Index(index_like)
    if not isinstance(idx, pd.MultiIndex):
        idx = idx[~idx.isin(["error", "skipped"])]
        num = pd.to_numeric(idx, errors="coerce")
        valid = pd.Series(num).notna().to_numpy()
        idx = pd.Index(num[valid].astype(int))
    return idx


def _extract_visit_from_target(target_col: str) -> Optional[int]:
    m = _VISIT_SUFFIX_RE.search(str(target_col))
    return int(m.group(1)) if m else None


def _multiindex_id_visit_levels(index: pd.MultiIndex) -> Tuple[int, int]:
    names = [str(n).lower() if n is not None else "" for n in index.names]
    id_level = 0
    visit_level = 1 if index.nlevels > 1 else 0

    for i, n in enumerate(names):
        if n in {"id", "eid", "subject_id", "participant_id"}:
            id_level = i
        if "visit" in n:
            visit_level = i

    if id_level == visit_level and index.nlevels > 1:
        visit_level = 1 if id_level == 0 else 0
    return id_level, visit_level


def _select_rows_for_visit(df: pd.DataFrame, visit: Optional[int]) -> Tuple[pd.DataFrame, pd.Index]:
    work = df.copy()

    if isinstance(work.index, pd.MultiIndex):
        id_level, visit_level = _multiindex_id_visit_levels(work.index)
        visit_vals = work.index.get_level_values(visit_level)

        if visit is not None:
            visit_mask = visit_vals.astype(str) == str(visit)
            work = work.loc[visit_mask]

        ids = pd.Index(work.index.get_level_values(id_level))
        work = work.copy()
        work.index = ids
    else:
        work = work.copy()
        work.index = _clean_index_ids(work.index)

    work = work.loc[~work.index.isin(["error", "skipped"])]
    num_ids = pd.to_numeric(work.index, errors="coerce")
    valid = pd.Series(num_ids, index=work.index).notna()
    work = work.loc[valid.values].copy()
    work.index = pd.Index(num_ids[valid.values].astype(int))

    if work.index.has_duplicates:
        # Keep one row per subject id for grouped CV.
        # Use numeric-mean when possible, otherwise fallback to first row.
        numeric_work = work.apply(pd.to_numeric, errors="coerce")
        grouped_numeric = numeric_work.groupby(numeric_work.index).mean()
        if grouped_numeric.shape[1] > 0:
            work = grouped_numeric
        else:
            work = work.groupby(work.index).first()

    return work, pd.Index(work.index)


def _select_target_series_for_visit(targets_df: pd.DataFrame, target_col: str) -> Tuple[pd.Series, Optional[int]]:
    visit = _extract_visit_from_target(target_col)

    if isinstance(targets_df.index, pd.MultiIndex):
        id_level, visit_level = _multiindex_id_visit_levels(targets_df.index)
        sub = targets_df[[target_col]].copy()

        if visit is not None:
            visit_vals = sub.index.get_level_values(visit_level).astype(str)
            sub = sub.loc[visit_vals == str(visit)]

        ids = pd.Index(sub.index.get_level_values(id_level))
        y = pd.Series(sub[target_col].values, index=ids, name=target_col)
    else:
        y = targets_df[target_col].copy()

    y = y.dropna()
    y = y.loc[~y.index.isin(["error", "skipped"])]
    num_ids = pd.to_numeric(y.index, errors="coerce")
    valid = pd.Series(num_ids, index=y.index).notna()
    y = y.loc[valid.values].copy()
    y.index = pd.Index(num_ids[valid.values].astype(int))

    if y.index.has_duplicates:
        y = y.groupby(y.index).first()

    return y, visit


def _is_classification_target(y: pd.Series, classification_max_unique: int = 10) -> bool:
    if y.dtype == bool:
        return True
    
    # -> FIXED LINE BELOW <-
    if pd.api.types.is_object_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
        return True

    nunique = y.nunique(dropna=True)
    if nunique <= classification_max_unique:
        # Keep small-integer targets as classification.
        vals = pd.to_numeric(y, errors="coerce")
        if vals.notna().mean() > 0.95:
            return np.allclose(vals.dropna(), np.round(vals.dropna()))
        return True

    return False

    nunique = y.nunique(dropna=True)
    if nunique <= classification_max_unique:
        # Keep small-integer targets as classification.
        vals = pd.to_numeric(y, errors="coerce")
        if vals.notna().mean() > 0.95:
            return np.allclose(vals.dropna(), np.round(vals.dropna()))
        return True

    return False


def _shuffled_group_kfold_split(
    groups: np.ndarray,
    n_splits: int,
    random_seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq_groups = pd.Index(groups).drop_duplicates().to_numpy()
    if len(uniq_groups) < n_splits:
        return []

    rng = np.random.RandomState(random_seed)
    shuffled = uniq_groups.copy()
    rng.shuffle(shuffled)
    group_folds = np.array_split(shuffled, n_splits)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_groups in group_folds:
        if len(fold_groups) == 0:
            continue
        te_mask = np.isin(groups, fold_groups)
        tr_idx = np.where(~te_mask)[0]
        te_idx = np.where(te_mask)[0]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        splits.append((tr_idx, te_idx))
    return splits


def _make_group_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    random_seed: int,
    is_classification: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if is_classification and StratifiedGroupKFold is not None:
        try:
            splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            return list(splitter.split(X, y, groups=groups))
        except Exception:
            pass

    return _shuffled_group_kfold_split(groups=groups, n_splits=n_folds, random_seed=random_seed)


def _fit_predict_single_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    is_classification: bool,
    alpha: float,
) -> Tuple[float, float]:
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if is_classification:
        model = RidgeClassifier(alpha=float(alpha))
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        return float(accuracy_score(y_test, y_pred)), np.nan

    # Regression
    model = Ridge(alpha=float(alpha))
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    if len(y_test) < 2 or np.nanstd(y_test) == 0 or np.nanstd(y_pred) == 0:
        return np.nan, np.nan

    score, _ = pearsonr(y_test, y_pred)
    return float(score), float(r2_score(y_test, y_pred))


def _search_best_alpha_non_nested(
    X_values: np.ndarray,
    y_values: np.ndarray,
    groups: np.ndarray,
    is_classification: bool,
    random_seed: int,
    n_iter_search: int,
) -> float:
    # Non-nested search: choose alpha once per seed using one group-aware tune split.
    tune_splits = _shuffled_group_kfold_split(groups=groups, n_splits=5, random_seed=random_seed)
    if not tune_splits:
        return 1.0

    tr_idx, va_idx = tune_splits[0]
    X_tr, X_va = X_values[tr_idx], X_values[va_idx]
    y_tr, y_va = y_values[tr_idx], y_values[va_idx]

    if len(X_tr) == 0 or len(X_va) == 0:
        return 1.0

    # Sample alphas log-uniformly over [1e-4, 1e2].
    rng = np.random.RandomState(random_seed)
    alphas = 10 ** rng.uniform(-4.0, 2.0, size=max(2, int(n_iter_search)))

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    best_alpha = float(alphas[0])
    best_score = -np.inf

    for a in alphas:
        try:
            if is_classification:
                if len(np.unique(y_tr)) < 2:
                    continue
                model = RidgeClassifier(alpha=float(a))
                model.fit(X_tr_s, y_tr)
                pred = model.predict(X_va_s)
                score = float(accuracy_score(y_va, pred))
            else:
                model = Ridge(alpha=float(a))
                model.fit(X_tr_s, y_tr)
                pred = model.predict(X_va_s)
                if len(y_va) < 2 or np.nanstd(y_va) == 0 or np.nanstd(pred) == 0:
                    score = -np.inf
                else:
                    score, _ = pearsonr(y_va, pred)
                    score = float(score)
            if score > best_score:
                best_score = score
                best_alpha = float(a)
        except Exception:
            continue

    return best_alpha


def evaluate_tabular_targets_multi_seed(
    targets_df: pd.DataFrame,
    baseline_features_df: pd.DataFrame,
    embeddings_dict: Dict[str, pd.DataFrame],
    min_samples: int = 100,
    classification_max_unique: int = 10,
    n_folds: int = 5,
    n_iter_search: int = 5,
    cv_search: int = 2,
    n_seeds: int = 5,
    seeds: Optional[Sequence[int]] = None,
    baseline_name: str = "Baseline",
) -> pd.DataFrame:
    if targets_df is None or baseline_features_df is None:
        raise ValueError("Provide targets_df and baseline_features_df.")
    if not embeddings_dict:
        raise ValueError("embeddings_dict must contain at least one embedding dataframe.")

    use_seeds = list(seeds) if seeds is not None else _default_seeds(n_seeds=n_seeds, base=73)
    sources: Dict[str, pd.DataFrame] = {baseline_name: baseline_features_df}
    for name, df in embeddings_dict.items():
        sources[str(name)] = df

    records: List[Dict[str, object]] = []
    candidate_targets = [
        c for c in targets_df.columns
        if (not targets_df[c].isna().all()) and (targets_df[c].notna().sum() >= min_samples)
    ]

    for target_col in candidate_targets:
        y_series, target_visit = _select_target_series_for_visit(targets_df, target_col)
        if y_series.empty:
            continue

        for source_name, source_df in sources.items():
            X_df, source_ids = _select_rows_for_visit(source_df, target_visit)
            common_ids = y_series.index.intersection(source_ids)

            if len(common_ids) < min_samples:
                continue

            y = y_series.loc[common_ids]
            if y.nunique(dropna=True) <= 1:
                continue

            X = X_df.loc[common_ids].copy()
            if X.empty:
                continue

            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan)

            valid_row_mask = ~X.isna().all(axis=1)
            X = X.loc[valid_row_mask]
            y = y.loc[valid_row_mask]
            if len(y) < min_samples:
                continue

            col_nan_fraction = X.isna().mean(axis=0)
            X = X.loc[:, col_nan_fraction < 0.95]
            if X.shape[1] == 0:
                continue

            X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
            groups = X.index.to_numpy()
            X_values = X.to_numpy(dtype=float)

            is_classification = _is_classification_target(y, classification_max_unique=classification_max_unique)

            if is_classification:
                # ---> ADD THIS BLOCK TO DYNAMICALLY DROP RARE CLASSES <---
                y_counts = y.value_counts()
                valid_classes = y_counts[y_counts >= n_folds].index
                
                # If we don't even have 2 valid classes left after intersecting with embeddings, skip target
                if len(valid_classes) < 2:
                    continue
                
                # Filter both X and y to only include the valid classes
                valid_mask = y.isin(valid_classes)
                y = y.loc[valid_mask]
                X = X.loc[valid_mask]
                
                # Re-sync the numpy arrays
                groups = X.index.to_numpy()
                X_values = X.to_numpy(dtype=float)

                y_raw = y.astype(str)
                y_encoded = LabelEncoder().fit_transform(y_raw)
                metric_name = "Accuracy"
                task_name = "classification"
                y_values = y_encoded
            else:
                y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(y_values)
                X_values = X_values[ok]
                groups = groups[ok]
                y_values = y_values[ok]
                metric_name = "Pearson r"
                task_name = "regression"

            if len(y_values) < min_samples:
                continue

            for seed in use_seeds:
                fold_scores: List[float] = []
                fold_r2s: List[float] = []

                best_alpha = _search_best_alpha_non_nested(
                    X_values=X_values,
                    y_values=y_values,
                    groups=groups,
                    is_classification=is_classification,
                    random_seed=int(seed),
                    n_iter_search=n_iter_search,
                )

                splits = _make_group_splits(
                    X=X_values,
                    y=y_values,
                    groups=groups,
                    n_folds=n_folds,
                    random_seed=int(seed),
                    is_classification=is_classification,
                )
                if not splits:
                    continue

                for train_idx, test_idx in splits:
                    X_train, X_test = X_values[train_idx], X_values[test_idx]
                    y_train, y_test = y_values[train_idx], y_values[test_idx]

                    if is_classification:
                        # Need at least two classes in train for classifier fitting.
                        if len(np.unique(y_train)) < 2:
                            continue

                    try:
                        score, fold_r2 = _fit_predict_single_fold(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            is_classification=is_classification,
                            alpha=best_alpha,
                        )
                    except Exception:
                        score, fold_r2 = np.nan, np.nan

                    fold_scores.append(score)
                    fold_r2s.append(fold_r2)

                if len(fold_scores) == 0:
                    continue

                records.append(
                    {
                        "Seed": int(seed),
                        "Target": target_col,
                        "Visit": target_visit,
                        "Source": source_name,
                        "Task": task_name,
                        "Metric": metric_name,
                        "N": int(len(y_values)),
                        "Score": float(np.nanmean(fold_scores)),
                        "Score SD (folds)": float(np.nanstd(fold_scores)),
                        "R2": float(np.nanmean(fold_r2s)) if task_name == "regression" else np.nan,
                    }
                )

    return pd.DataFrame.from_records(records, columns=_SEED_LEVEL_COLUMNS)


def summarize_seed_results(seed_level_df: pd.DataFrame) -> pd.DataFrame:
    if seed_level_df.empty:
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)

    rows: List[Dict[str, object]] = []
    keys = ["Target", "Visit", "Source", "Task", "Metric"]

    for key, grp in seed_level_df.groupby(keys, dropna=False):
        scores = pd.to_numeric(grp["Score"], errors="coerce")
        valid = scores[np.isfinite(scores)]
        n_valid = int(len(valid))
        if n_valid == 0:
            continue

        se = float(np.std(valid, ddof=1) / np.sqrt(n_valid)) if n_valid > 1 else 0.0
        rows.append(
            {
                "Target": key[0],
                "Visit": key[1],
                "Source": key[2],
                "Task": key[3],
                "Metric": key[4],
                "Mean Score": float(np.mean(valid)),
                "SE Score": se,
                "N Seeds": n_valid,
                "N": int(pd.to_numeric(grp["N"], errors="coerce").median()),
                "Mean R2": float(pd.to_numeric(grp["R2"], errors="coerce").mean()) if key[3] == "regression" else np.nan,
            }
        )

    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


def paired_ttests_vs_baseline(
    seed_level_df: pd.DataFrame,
    baseline_name: str = "Baseline",
) -> pd.DataFrame:
    if seed_level_df.empty:
        return pd.DataFrame(columns=_PVAL_COLUMNS)

    out_rows: List[Dict[str, object]] = []
    pair_keys = ["Target", "Visit", "Task", "Metric"]

    for pair_key, grp in seed_level_df.groupby(pair_keys, dropna=False):
        base_grp = grp[grp["Source"] == baseline_name]
        if base_grp.empty:
            continue

        base_seed_score = base_grp.set_index("Seed")["Score"]

        for source_name, s_grp in grp.groupby("Source"):
            if source_name == baseline_name:
                continue

            src_seed_score = s_grp.set_index("Seed")["Score"]
            common_seeds = base_seed_score.index.intersection(src_seed_score.index)
            if len(common_seeds) < 2:
                p_val = np.nan
                t_stat = np.nan
                n_pairs = int(len(common_seeds))
            else:
                a = pd.to_numeric(src_seed_score.loc[common_seeds], errors="coerce")
                b = pd.to_numeric(base_seed_score.loc[common_seeds], errors="coerce")
                valid_mask = np.isfinite(a) & np.isfinite(b)
                a = a[valid_mask]
                b = b[valid_mask]
                n_pairs = int(len(a))
                if n_pairs < 2:
                    p_val = np.nan
                    t_stat = np.nan
                elif np.allclose(a.values, b.values):
                    p_val = 1.0
                    t_stat = 0.0
                else:
                    t_stat, p_val = ttest_rel(a.values, b.values, nan_policy="omit")

            out_rows.append(
                {
                    "Target": pair_key[0],
                    "Visit": pair_key[1],
                    "Task": pair_key[2],
                    "Metric": pair_key[3],
                    "Baseline": baseline_name,
                    "Source": source_name,
                    "N Pairs": n_pairs,
                    "T Statistic": float(t_stat) if pd.notna(t_stat) else np.nan,
                    "P Value": float(p_val) if pd.notna(p_val) else np.nan,
                }
            )

    return pd.DataFrame(out_rows, columns=_PVAL_COLUMNS)


def _significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_bar_comparison_with_pvalues(
    summary_df: pd.DataFrame,
    pvals_df: pd.DataFrame,
    baseline_name: str = "Baseline",
    out_dir: str = ".",
    show: bool = True,
) -> List[Path]:
    saved_paths: List[Path] = []
    if summary_df.empty:
        return saved_paths

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for (task, metric), sub in summary_df.groupby(["Task", "Metric"], dropna=False):
        work = sub.copy()
        if work.empty:
            continue

        # Stable ordering by baseline then best score.
        base_scores = (
            work[work["Source"] == baseline_name][["Target", "Mean Score"]]
            .rename(columns={"Mean Score": "BaseScore"})
            .set_index("Target")
        )
        work = work.join(base_scores, on="Target")
        work["SortScore"] = work["BaseScore"].fillna(work["Mean Score"])
        target_order = (
            work.groupby("Target")["SortScore"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )

        sources = work["Source"].drop_duplicates().tolist()
        n_sources = len(sources)
        x = np.arange(len(target_order), dtype=float)
        width = 0.8 / max(1, n_sources)

        fig_w = max(14.0, 0.45 * len(target_order) + 4.0)
        fig, ax = plt.subplots(figsize=(fig_w, 7.0))

        source_to_offset: Dict[str, float] = {}
        for i, src in enumerate(sources):
            offset = (i - (n_sources - 1) / 2.0) * width
            source_to_offset[src] = offset

            src_sub = work[work["Source"] == src].set_index("Target").reindex(target_order)
            vals = pd.to_numeric(src_sub["Mean Score"], errors="coerce").to_numpy()
            errs = pd.to_numeric(src_sub["SE Score"], errors="coerce").fillna(0.0).to_numpy()

            ax.bar(
                x + offset,
                vals,
                width=width,
                yerr=errs,
                capsize=3,
                label=src,
                alpha=0.9,
                zorder=3,
            )

        baseline_line = 0.5 if task == "regression" else None
        if baseline_line is not None:
            ax.axhline(baseline_line, color="#7F8C8D", linestyle=":", linewidth=1.5, zorder=1)

        if not pvals_df.empty:
            p_sub = pvals_df[(pvals_df["Task"] == task) & (pvals_df["Metric"] == metric)]
            for _, row in p_sub.iterrows():
                src = row["Source"]
                tgt = row["Target"]
                p_val = row["P Value"]
                if src not in source_to_offset or tgt not in target_order:
                    continue

                stars = _significance_stars(p_val)
                if not stars:
                    continue

                x0 = x[target_order.index(tgt)] + source_to_offset[src]
                src_row = work[(work["Source"] == src) & (work["Target"] == tgt)]
                if src_row.empty:
                    continue
                y0 = float(src_row["Mean Score"].iloc[0]) + float(src_row["SE Score"].fillna(0.0).iloc[0]) + 0.01
                ax.text(x0, y0, stars, ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(target_order, rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{task.title()} comparison across sources")
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.legend(frameon=False, loc="upper left")
        plt.tight_layout()

        file_path = out_path / f"downstream_bar_{task}_{metric.replace(' ', '_').lower()}.png"
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        saved_paths.append(file_path)

        if show:
            plt.show()
        plt.close(fig)

    return saved_paths


def plot_radar_comparison_with_pvalues(
    summary_df: pd.DataFrame,
    pvals_df: pd.DataFrame,
    baseline_name: str = "Baseline",
    out_dir: str = ".",
    show: bool = True,
) -> List[Path]:
    saved_paths: List[Path] = []
    if summary_df.empty:
        return saved_paths

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    non_baseline_sources = [s for s in summary_df["Source"].drop_duplicates().tolist() if s != baseline_name]

    for source_name in non_baseline_sources:
        for (task, metric), sub in summary_df.groupby(["Task", "Metric"], dropna=False):
            pair = sub[sub["Source"].isin([baseline_name, source_name])].copy()
            if pair.empty:
                continue

            mat = pair.pivot_table(index="Target", columns="Source", values="Mean Score", aggfunc="mean")
            if baseline_name not in mat.columns or source_name not in mat.columns:
                continue
            mat = mat[[baseline_name, source_name]].dropna()
            if len(mat) < 3:
                continue

            p_sub = pvals_df[
                (pvals_df["Task"] == task)
                & (pvals_df["Metric"] == metric)
                & (pvals_df["Source"] == source_name)
            ][["Target", "P Value"]].set_index("Target")

            labels = mat.index.tolist()
            display_labels = []
            for t in labels:
                p = p_sub["P Value"].get(t, np.nan) if not p_sub.empty else np.nan
                display_labels.append(f"{t}{_significance_stars(p)}")

            n = len(labels)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
            angles += angles[:1]

            vals_base = mat[baseline_name].tolist()
            vals_src = mat[source_name].tolist()
            vals_base += vals_base[:1]
            vals_src += vals_src[:1]

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
            ax.plot(angles, vals_base, linewidth=2.0, label=baseline_name)
            ax.fill(angles, vals_base, alpha=0.12)
            ax.plot(angles, vals_src, linewidth=2.0, label=source_name)
            ax.fill(angles, vals_src, alpha=0.12)

            ax.set_thetagrids(np.degrees(angles[:-1]), display_labels)
            ax.tick_params(axis="x", labelsize=8)

            all_vals = vals_base[:-1] + vals_src[:-1]
            r_min = min(0.45, float(np.nanmin(all_vals) - 0.03))
            r_max = min(1.0, float(np.nanmax(all_vals) + 0.05))
            if r_max <= r_min:
                r_min, r_max = 0.45, 1.0
            ax.set_ylim(r_min, r_max)

            if task == "regression":
                ax.plot(angles, [0.5] * len(angles), color="#7F8C8D", linestyle=":", linewidth=1.5)

            ax.set_title(f"{task.title()} radar: {source_name} vs {baseline_name} ({metric})")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
            plt.tight_layout()

            file_path = out_path / f"downstream_radar_{task}_{metric.replace(' ', '_').lower()}_{source_name}.png"
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
            saved_paths.append(file_path)

            if show:
                plt.show()
            plt.close(fig)

    return saved_paths


def run_downstream_comparison(
    targets_df: pd.DataFrame,
    baseline_features_df: pd.DataFrame,
    embeddings_dict: Dict[str, pd.DataFrame],
    output_dir: str,
    baseline_name: str = "Baseline",
    min_samples: int = 100,
    classification_max_unique: int = 10,
    n_folds: int = 5,
    n_iter_search: int = 5,
    cv_search: int = 2,
    n_seeds: int = 5,
    seeds: Optional[Sequence[int]] = None,
    show_plots: bool = True,
) -> Dict[str, object]:
    seed_level_df = evaluate_tabular_targets_multi_seed(
        targets_df=targets_df,
        baseline_features_df=baseline_features_df,
        embeddings_dict=embeddings_dict,
        min_samples=min_samples,
        classification_max_unique=classification_max_unique,
        n_folds=n_folds,
        n_iter_search=n_iter_search,
        cv_search=cv_search,
        n_seeds=n_seeds,
        seeds=seeds,
        baseline_name=baseline_name,
    )

    summary_df = summarize_seed_results(seed_level_df)
    pvals_df = paired_ttests_vs_baseline(seed_level_df, baseline_name=baseline_name)

    if not summary_df.empty and not pvals_df.empty:
        summary_df = summary_df.merge(
            pvals_df[["Target", "Visit", "Task", "Metric", "Source", "P Value", "N Pairs"]],
            on=["Target", "Visit", "Task", "Metric", "Source"],
            how="left",
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_csv = out_dir / "downstream_seed_level_results.csv"
    summary_csv = out_dir / "downstream_summary_with_pvalues.csv"
    pvals_csv = out_dir / "downstream_paired_ttests_vs_baseline.csv"

    seed_level_df.to_csv(seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    pvals_df.to_csv(pvals_csv, index=False)

    bar_paths = plot_bar_comparison_with_pvalues(
        summary_df=summary_df,
        pvals_df=pvals_df,
        baseline_name=baseline_name,
        out_dir=str(out_dir),
        show=show_plots,
    )
    radar_paths = plot_radar_comparison_with_pvalues(
        summary_df=summary_df,
        pvals_df=pvals_df,
        baseline_name=baseline_name,
        out_dir=str(out_dir),
        show=show_plots,
    )

    return {
        "seed_level": seed_level_df,
        "summary": summary_df,
        "pvalues": pvals_df,
        "seed_csv": str(seed_csv),
        "summary_csv": str(summary_csv),
        "pvalues_csv": str(pvals_csv),
        "bar_paths": [str(p) for p in bar_paths],
        "radar_paths": [str(p) for p in radar_paths],
    }


def _read_table(path: str, index_col: Optional[int] = 0) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p, index_col=index_col, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(p)
    raise ValueError(f"Unsupported file format: {path}")


def _set_eid_index_if_present(df: pd.DataFrame) -> pd.DataFrame:
    # Auto-detect common eid column variants and use as index.
    for col in df.columns:
        c = str(col).strip().lower()
        if c == "eid":
            out = df.copy()
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out.dropna(subset=[col])
            out[col] = out[col].astype(int)
            out = out.set_index(col)
            return out
    return df


def _load_targets(path: str) -> pd.DataFrame:
    # Try wide format first; fallback to multiindex csv with eid/visit columns.
    df = _read_table(path, index_col=0)
    if isinstance(df.index, pd.MultiIndex):
        return df

    if "eid" in df.columns and "visit" in df.columns:
        out = df.copy()
        out["eid"] = pd.to_numeric(out["eid"], errors="coerce")
        out["visit"] = pd.to_numeric(out["visit"], errors="coerce")
        out = out.dropna(subset=["eid", "visit"])
        out["eid"] = out["eid"].astype(int)
        out["visit"] = out["visit"].astype(int)
        out = out.set_index(["eid", "visit"])
        return out

    df = _set_eid_index_if_present(df)

    return df


def _load_features(path: str) -> pd.DataFrame:
    df = _read_table(path, index_col=0)
    if "eid" in df.columns and "visit" in df.columns:
        out = df.copy()
        out["eid"] = pd.to_numeric(out["eid"], errors="coerce")
        out["visit"] = pd.to_numeric(out["visit"], errors="coerce")
        out = out.dropna(subset=["eid", "visit"])
        out["eid"] = out["eid"].astype(int)
        out["visit"] = out["visit"].astype(int)
        out = out.set_index(["eid", "visit"])
        return out

    df = _set_eid_index_if_present(df)
    return df


def _main() -> None:
    # Edit these paths once, then run this file directly.
    targets_path = "/home/gilsa/PycharmProjects/DEXA/dexa_fm_recovered_from_vscode/ukbb/ukbb_targets_slim.csv"
    baseline_path = "/home/gilsa/PycharmProjects/DEXA/ukbb_tabular_data_for_cox.csv"
    output_dir = "/home/gilsa/PycharmProjects/DEXA/dexa_fm_recovered_from_vscode/ukbb/downstream_results"

    embeddings_paths: Dict[str, str] = {
        "LeJEPA": "/home/gilsa/PycharmProjects/DEXA/lejepa_embeddings_for_cox.csv",
        "DINOv3": "/home/gilsa/PycharmProjects/DEXA/dino_embeddings_for_cox.csv",
    }

    # Fast defaults; increase n_folds/n_seeds for full runs.
    min_samples = 500
    n_folds = 3
    n_seeds = 5
    seeds: Optional[Sequence[int]] = None
    show_plots = True

    targets_df = _load_targets(targets_path)

    # ---> ADD THIS BLOCK TO FILTER FOR VISIT 3 <---
    # Keep only columns that explicitly have "- visit 3" in the name
    v3_cols = [c for c in targets_df.columns if "- visit 3" in str(c).lower()]
    # v2_cols = [c for c in targets_df.columns if "- visit 2" in str(c).lower()]
    targets_df = targets_df[v3_cols].copy()
    print(f"Filtered to {len(v3_cols)} Visit 3 targets.")
    exclude_keywords = [
        'date', 'month', 'year', 'assessment centre', 
        'ecg', 'axis', 'interval', 'duration', 'qrs',
        'device', 'batch', 'aliquot', 'operator', 'time of', 'ECG'
    ]

    clean_cols = []
    for col in targets_df.columns:
        col_lower = str(col).lower()
        # Keep the column ONLY if none of the forbidden words are in the name
        if not any(kw in col_lower for kw in exclude_keywords):
            clean_cols.append(col)

    targets_df = targets_df[clean_cols].copy()
    print(f"Filtered out admin/noise. {len(clean_cols)} clean clinical targets remain.")
    # targets_df = targets_df.iloc[:, :500]  # Limit to first 500 targets for testing.
    print(f"Limiting run to {targets_df.shape[1]} targets for testing.")
    # -----------------------------------------------

    baseline_df = _load_features(baseline_path)
    embeddings_dict = {name: _load_features(path) for name, path in embeddings_paths.items()}

    if targets_df.shape[1] == 0:
        raise ValueError(
            f"No target columns loaded from {targets_path}. "
            "Check that the file contains phenotype columns in addition to eid/visit columns."
        )

    print("Loaded shapes:")
    print("  targets:", targets_df.shape)
    print("  baseline:", baseline_df.shape)
    for name, emb_df in embeddings_dict.items():
        print(f"  {name}:", emb_df.shape)

    target_ids = pd.Index(pd.to_numeric(targets_df.index, errors="coerce")).dropna().astype(int).unique()
    base_ids = pd.Index(pd.to_numeric(baseline_df.index, errors="coerce")).dropna().astype(int).unique()
    print("ID overlap targets-baseline:", len(pd.Index(target_ids).intersection(pd.Index(base_ids))))
    for name, emb_df in embeddings_dict.items():
        emb_ids = pd.Index(pd.to_numeric(emb_df.index, errors="coerce")).dropna().astype(int).unique()
        print(f"ID overlap targets-{name}:", len(pd.Index(target_ids).intersection(pd.Index(emb_ids))))

    eligible_targets = int((targets_df.notna().sum(axis=0) >= min_samples).sum())
    print(f"Targets with N >= {min_samples}:", eligible_targets)

    result = run_downstream_comparison(
        targets_df=targets_df,
        baseline_features_df=baseline_df,
        embeddings_dict=embeddings_dict,
        output_dir=output_dir,
        baseline_name="Baseline",
        min_samples=min_samples,
        classification_max_unique=10,
        n_folds=n_folds,
        n_iter_search=5,
        cv_search=2,
        n_seeds=n_seeds,
        seeds=seeds,
        show_plots=show_plots,
    )

    print("Completed downstream comparison.")
    print("Seed-level CSV:", result["seed_csv"])
    print("Summary CSV:", result["summary_csv"])
    print("Paired t-tests CSV:", result["pvalues_csv"])
    print("Bar plots:", len(result["bar_paths"]))
    print("Radar plots:", len(result["radar_paths"]))


if __name__ == "__main__":
    _main()