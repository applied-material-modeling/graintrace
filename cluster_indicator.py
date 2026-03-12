from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Callable, Dict, Any
from user_data_class import SimilarityMetric


class ClusterAnalysisIndicator:
    def __init__(
        self,
        csv_path: str,
        id_col: str = "id",
        coord_cols: Tuple[str, str, str] = ("x", "y", "z"),
    ):
        self.csv_path = csv_path
        self.id_col = id_col
        self.coord_cols = coord_cols

        self.data: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Load CSV and populate data, features, coords."""
        if self.data is not None:
            return

        df = pd.read_csv(self.csv_path)

        # check essential columns
        required = [self.id_col, *self.coord_cols]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # if there are no other columns other than required, raise error
        if len(df.columns) == len(required):
            raise ValueError("No feature columns found in the data.")

        self.data = df

    def check_feature_matrix(self, spec: SimilarityMetric) -> None:

        if self.data is None:
            self.load_data()

        df = self.data

        missing = [c for c in spec.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Metric '{spec.name}' requires missing columns: {missing}"
            )

    def _build_cluster_summaries_from_arrays(
        self,
        labels: np.ndarray,
        coords: np.ndarray,
        feats: np.ndarray,
        coord_names: List[str],
        feat_names: List[str],
        include_noise: bool = False,
        noise_label: int = -1,
        label_col: str = "cluster_label",
    ) -> pd.DataFrame:
        if labels.ndim != 1:
            labels = labels.ravel()

        if not include_noise:
            m = labels != noise_label
            labels = labels[m]
            coords = coords[m]
            feats = feats[m]

        if labels.size == 0:
            cols = (
                [label_col, "n"]
                + [f"{c}_min" for c in coord_names]
                + [f"{c}_max" for c in coord_names]
                + [f"{c}_sum" for c in coord_names]
                + [f"{c}_sumsq" for c in coord_names]
                + [f"{f}_sum" for f in feat_names]
                + [f"{f}_sumsq" for f in feat_names]
                + [f"{c}_mean" for c in coord_names]
                + [f"{f}_mean" for f in feat_names]
            )
            return pd.DataFrame(columns=cols)

        uniq, inv = np.unique(labels, return_inverse=True)
        k = uniq.size

        n = np.bincount(inv, minlength=k).astype(np.int64)

        def _bincount_cols(mat: np.ndarray, power: int = 1) -> np.ndarray:
            out = np.empty((k, mat.shape[1]), dtype=np.float64)
            for j in range(mat.shape[1]):
                col = mat[:, j]
                if power == 2:
                    col = col * col
                out[:, j] = np.bincount(inv, weights=col, minlength=k)
            return out

        coord_sum = _bincount_cols(coords, power=1)
        coord_sumsq = _bincount_cols(coords, power=2)
        feat_sum = _bincount_cols(feats, power=1)
        feat_sumsq = _bincount_cols(feats, power=2)

        order = np.argsort(inv, kind="mergesort")
        inv_s = inv[order]
        coords_s = coords[order]

        starts = np.flatnonzero(np.r_[True, inv_s[1:] != inv_s[:-1]])
        # starts length == k
        coord_min = np.empty((k, coords.shape[1]), dtype=np.float64)
        coord_max = np.empty((k, coords.shape[1]), dtype=np.float64)
        for j in range(coords.shape[1]):
            col = coords_s[:, j]
            coord_min[:, j] = np.minimum.reduceat(col, starts)
            coord_max[:, j] = np.maximum.reduceat(col, starts)

        # Assemble summaries
        data: Dict[str, Any] = {label_col: uniq, "n": n}

        for j, c in enumerate(coord_names):
            data[f"{c}_min"] = coord_min[:, j]
            data[f"{c}_max"] = coord_max[:, j]
            data[f"{c}_sum"] = coord_sum[:, j]
            data[f"{c}_sumsq"] = coord_sumsq[:, j]
            data[f"{c}_mean"] = coord_sum[:, j] / n

        for j, f in enumerate(feat_names):
            data[f"{f}_sum"] = feat_sum[:, j]
            data[f"{f}_sumsq"] = feat_sumsq[:, j]
            data[f"{f}_mean"] = feat_sum[:, j] / n

        return pd.DataFrame(data)

    def _get_all_feature_cols(self, df: pd.DataFrame) -> List[str]:
        required = [self.id_col, *self.coord_cols]
        return [c for c in df.columns if c not in required]

    def run(
        self,
        method_type: str,
        spec: SimilarityMetric,
        minimal_return: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Returns dict:
          {
            "points": labeled_points_df,
            "clusters": cluster_summaries_df,
            "extras": {...}  # method-specific, e.g. linkage Z
          }
        """
        self.load_data()
        self.check_feature_matrix(spec)

        if method_type == "scipy_hierarchical":
            points, clusters, extras = self.run_scipy_hierarchical(spec, **kwargs)
        elif method_type == "sklearn_dbscan":
            points, clusters, extras = self.run_sklearn_dbscan(spec, **kwargs)
        elif method_type == "sklearn_agglomerative":
            points, clusters, extras = self.run_sklearn_agglomerative(spec, **kwargs)
        elif method_type == "sklearn_optics":
            points, clusters, extras = self.run_sklearn_optics(spec, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method_type}")

        if minimal_return:
            return {"clusters": clusters}

        return {"points": points, "clusters": clusters, "extras": extras}

    ## different clustering methods
    def run_sklearn_dbscan(
        self,
        spec: SimilarityMetric,
        eps: float = 0.5,
        min_samples: int = 5,
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: Optional[float] = None,
        n_jobs: Optional[int] = None,
        include_noise_in_summaries: bool = False,
        noise_label: int = -1,
        minimal_return: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:

        if self.data is None:
            self.load_data()
        df = self.data

        X = df[spec.feature_cols].to_numpy(dtype=float)
        coords = df[list(self.coord_cols)].to_numpy(dtype=float)

        all_feat_cols = self._get_all_feature_cols(df)
        X_all = df[all_feat_cols].to_numpy(dtype=float)

        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=spec.func,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        ).fit(X)

        labels = clustering.labels_

        clusters = self._build_cluster_summaries_from_arrays(
            labels=labels,
            coords=coords,
            feats=X_all,
            coord_names=list(self.coord_cols),
            feat_names=list(spec.feature_cols),
            include_noise=include_noise_in_summaries,
            noise_label=noise_label,
            label_col="cluster_label",
        )

        extras = {
            "n_clusters_excluding_noise": int(
                len(set(labels)) - (1 if noise_label in labels else 0)
            ),
            "n_noise": int(np.sum(labels == noise_label)),
        }

        if minimal_return:
            return None, clusters, extras

        points = df.copy()
        points["cluster_label"] = labels
        return points, clusters, extras

    def run_sklearn_agglomerative(
        self,
        spec: SimilarityMetric,
        n_clusters: Optional[int] = 2,
        memory: Optional[Any] = None,
        connectivity: Optional[Any] = None,
        compute_full_tree: Any = "auto",
        linkage: str = "average",
        distance_threshold: Optional[float] = None,
        compute_distances: bool = False,
        minimal_return: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:

        if self.data is None:
            self.load_data()
        df = self.data

        if linkage == "ward":
            raise ValueError(
                "AgglomerativeClustering with linkage='ward' does not support "
                "a callable metric. Use 'average', 'complete', or 'single'."
            )

        X = df[spec.feature_cols].to_numpy(dtype=float)
        coords = df[list(self.coord_cols)].to_numpy(dtype=float)

        all_feat_cols = self._get_all_feature_cols(df)
        X_all = df[all_feat_cols].to_numpy(dtype=float)

        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=spec.func,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances,
        )

        labels = clustering.fit_predict(X)

        clusters = self._build_cluster_summaries_from_arrays(
            labels=labels,
            coords=coords,
            feats=X_all,
            coord_names=list(self.coord_cols),
            feat_names=all_feat_cols,
            include_noise=True,  # agglomerative has no noise label
            noise_label=-1,
            label_col="cluster_label",
        )

        extras = {"n_clusters": int(len(np.unique(labels)))}

        if minimal_return:
            return None, clusters, extras

        points = df.copy()
        points["cluster_label"] = labels
        return points, clusters, extras

    def run_sklearn_optics(
        self,
        spec: SimilarityMetric,
        min_samples: Any = 5,
        max_eps: float = np.inf,
        p: float = 2,
        metric_params: Optional[Dict[str, Any]] = None,
        cluster_method: str = "xi",
        eps: Optional[float] = None,
        xi: float = 0.05,
        predecessor_correction: bool = True,
        min_cluster_size: Optional[int] = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        memory: Optional[Any] = None,
        n_jobs: Optional[int] = None,
        include_noise_in_summaries: bool = False,
        noise_label: int = -1,
        minimal_return: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:

        if self.data is None:
            self.load_data()
        df = self.data

        X = df[spec.feature_cols].to_numpy(dtype=float)
        coords = df[list(self.coord_cols)].to_numpy(dtype=float)

        all_feat_cols = self._get_all_feature_cols(df)
        X_all = df[all_feat_cols].to_numpy(dtype=float)

        from sklearn.cluster import OPTICS

        clustering = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            metric=spec.func,
            p=p,
            metric_params=metric_params,
            cluster_method=cluster_method,
            eps=eps,
            xi=xi,
            predecessor_correction=predecessor_correction,
            min_cluster_size=min_cluster_size,
            algorithm=algorithm,
            leaf_size=leaf_size,
            memory=memory,
            n_jobs=n_jobs,
        ).fit(X)

        labels = clustering.labels_

        clusters = self._build_cluster_summaries_from_arrays(
            labels=labels,
            coords=coords,
            feats=X_all,
            coord_names=list(self.coord_cols),
            feat_names=all_feat_cols,
            include_noise=include_noise_in_summaries,
            noise_label=noise_label,
            label_col="cluster_label",
        )

        extras = {
            "n_clusters_excluding_noise": int(
                len(set(labels)) - (1 if noise_label in labels else 0)
            ),
            "n_noise": int(np.sum(labels == noise_label)),
        }

        if minimal_return:
            return None, clusters, extras

        points = df.copy()
        points["cluster_label"] = labels
        return points, clusters, extras

    def plot_dendrogram(
        self,
        Z: np.ndarray,
        threshold: float,
        save_path: Optional[str] = None,
        ax=None,
        no_labels: bool = True,
    ) -> Dict[str, Any]:

        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        else:
            fig = ax.figure

        dinfo = dendrogram(
            Z,
            color_threshold=threshold,
            ax=ax,
            no_labels=no_labels,
        )
        ax.axhline(y=threshold, color="k", linestyle="--")
        ax.set_ylabel("Ultrametric distance")

        if save_path is not None:
            fig.tight_layout()
            fig.savefig(save_path, dpi=300)

        return {
            "leaves": dinfo.get("leaves", []),
            "leaves_color_list": dinfo.get("leaves_color_list", []),
            "ivl": dinfo.get("ivl", []),
            "color_threshold": threshold,
        }

    def run_scipy_hierarchical(
        self,
        spec: SimilarityMetric,
        method: str = "average",
        criterion: str = "distance",
        threshold: float = 1.0,
        depth: int = 10,
        dendrogram_path: Optional[str] = None,
        minimal_return: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:

        if self.data is None:
            self.load_data()
        df = self.data

        X = df[spec.feature_cols].to_numpy(dtype=float)
        coords = df[list(self.coord_cols)].to_numpy(dtype=float)

        all_feat_cols = self._get_all_feature_cols(df)
        X_all = df[all_feat_cols].to_numpy(dtype=float)

        from scipy.cluster.hierarchy import linkage, cophenet, fclusterdata
        from scipy.spatial.distance import pdist, squareform
        from sklearn.manifold import MDS

        D = pdist(X, metric=spec.func)
        Z = linkage(X, method=method, metric=spec.func)

        labels = fclusterdata(
            X,
            t=threshold,
            criterion=criterion,
            metric=spec.func,
            depth=depth,
            method=method,
        )

        coph_corr, coph_dists = cophenet(Z, D)
        D_ultra = squareform(coph_dists)

        mds_1d = MDS(
            n_components=1, dissimilarity="precomputed", random_state=42, n_init=4
        )
        X_1d = mds_1d.fit_transform(D_ultra).ravel()

        clusters = self._build_cluster_summaries_from_arrays(
            labels=labels,
            coords=coords,
            feats=X_all,
            coord_names=list(self.coord_cols),
            feat_names=all_feat_cols,
            include_noise=True,
            noise_label=-1,
            label_col="cluster_label",
        )

        extras = {"linkage_Z": Z, "cophenetic_correlation": float(coph_corr)}

        if dendrogram_path is not None:
            extras["dendrogram"] = self.plot_dendrogram(
                Z=Z,
                threshold=threshold,
                save_path=dendrogram_path,
                ax=None,
                no_labels=True,
            )

        if minimal_return:
            return None, clusters, extras

        points = df.copy()
        points["cluster_label"] = labels
        points["mds_1d"] = X_1d
        return points, clusters, extras
