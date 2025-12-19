from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Callable, Dict, Any

DistanceFunction = Callable[[np.ndarray, np.ndarray], float]

@dataclass
class SimilarityMetric:
    name: str
    feature_cols: List[str]   # requried feature names
    func: DistanceFunction    # metric(u, v) -> float

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

    def run(
        self,
        method_type: str,
        spec: SimilarityMetric,
        **kwargs: Any,
    ) -> pd.DataFrame:

        # load data
        self.load_data()
        self.check_feature_matrix(spec)

        # run clustering based on method
        if method_type == "scipy_hierarchical":
            out = self.run_scipy_hierarchical(spec, **kwargs)
        elif method_type == "sklearn_dbscan":
            out = self.run_sklearn_dbscan(spec, **kwargs)
        elif method_type == "sklearn_agglomerative":
            out = self.run_sklearn_agglomerative(spec, **kwargs)
        elif method_type == "sklearn_optics":
            out = self.run_sklearn_optics(spec, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method_type}")

        # postprocessing and plots -- to be added later

        return out

    ## different clustering methods, returning the 
    def run_scipy_hierarchical(
        self,
        spec: SimilarityMetric,
        method: str = "average",
        criterion: str = "distance",
        threshold: float = 1.0,
        depth: int = 10
    ) -> pd.DataFrame:
        
        if self.data is None:
            self.load_data()

        df = self.data

        # reduce input data 
        data = df[spec.feature_cols].to_numpy() #np.ndarray of shape (n_samples, n_features)
        metric = spec.func

        # called clustering function
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, cophenet, fclusterdata
        from scipy.spatial.distance import pdist, squareform
        from sklearn.manifold import MDS

        # get linkage information for later usage
        D = pdist(data, metric=metric)
        Z = linkage(data, method=method, metric=metric)

        labels = fclusterdata(
            data,
            t=threshold,
            criterion=criterion,
            metric=metric,
            depth=depth,
            method=method,
        )

        coph_corr, coph_dists = cophenet(Z, D)   # coph_dists is condensed
        D_ultra = squareform(coph_dists)         # full ultrametric matrix

        mds_1d = MDS(n_components=1, dissimilarity="precomputed", random_state=42, n_init=4)
        X_1d = mds_1d.fit_transform(D_ultra).ravel()

        result = df.copy()
        result["cluster_label"] = labels
        result["mds_1d"] = X_1d

        return result, Z

    def run_sklearn_dbscan(
        self,
        spec: SimilarityMetric,
        eps: float = 0.5,
        min_samples: int = 5,
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: Optional[float] = None,
        n_jobs: Optional[int] = None,
    ) -> pd.DataFrame:

        if self.data is None:
            self.load_data()

        df = self.data

        data = df[spec.feature_cols].to_numpy(dtype=float)
        metric = spec.func

        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,            
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        ).fit(data)

        labels = clustering.labels_

        result = df.copy()
        result["cluster_label"] = labels
        return result

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
    ) -> pd.DataFrame:

        if self.data is None:
            self.load_data()

        df = self.data

        if linkage == "ward":
            raise ValueError(
                "AgglomerativeClustering with linkage='ward' does not support "
                "a callable metric. Use 'average', 'complete', or 'single'."
            )

        data = df[spec.feature_cols].to_numpy(dtype=float)
        metric = spec.func

        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,                # custom callable
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances,
        )

        labels = clustering.fit_predict(data)

        result = df.copy()
        result["cluster_label"] = labels
        return result

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
    ) -> pd.DataFrame:

        if self.data is None:
            self.load_data()

        df = self.data

        data = df[spec.feature_cols].to_numpy(dtype=float)
        metric = spec.func

        from sklearn.cluster import OPTICS

        clustering = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            metric=metric,                 # custom callable
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
        ).fit(data)

        labels = clustering.labels_

        result = df.copy()
        result["cluster_label"] = labels
        return result

    ## Plotting support functions