import numpy as np
from cluster_indicator import SimilarityMetric  # adjust import path

# von Mises stress distance
def von_mises_stress_distance(u: np.ndarray, v: np.ndarray) -> float:
    sxx_u, syy_u, szz_u, sxy_u, syz_u, sxz_u = u
    sxx_v, syy_v, szz_v, sxy_v, syz_v, sxz_v = v

    def von_mises(sxx, syy, szz, sxy, syz, sxz):
        term1 = ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) / 2
        term2 = 3 * (sxy ** 2 + syz ** 2 + sxz ** 2)
        return np.sqrt(term1 + term2)

    vm_u = von_mises(sxx_u, syy_u, szz_u, sxy_u, syz_u, sxz_u)
    vm_v = von_mises(sxx_v, syy_v, szz_v, sxy_v, syz_v, sxz_v)

    return float(abs(vm_u - vm_v) / (abs(vm_u) + abs(vm_v) + 1e-8))

def von_mises_stress_distance_batch(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    sxx = X[:, 0]; syy = X[:, 1]; szz = X[:, 2]
    sxy = X[:, 3]; syz = X[:, 4]; sxz = X[:, 5]
    term1 = ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) / 2.0
    term2 = 3.0 * (sxy**2 + syz**2 + sxz**2)
    vm = np.sqrt(term1 + term2)

    I = edges[:, 0]
    J = edges[:, 1]
    a = vm[I]
    b = vm[J]
    return np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8)

# misorientation - batch by default
def make_misorientation_dist_edges(
    angle_convention: str = "bunge",
    angle_type: str = "degrees",
    symmetry: str = "432",
):
    import torch
    from neml2 import tensors, crystallography

    def dist_edges(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
        I = edges[:, 0]
        J = edges[:, 1]

        e1 = torch.as_tensor(X[I], dtype=torch.float64)
        e2 = torch.as_tensor(X[J], dtype=torch.float64)

        e1 = tensors.Rot(e1)
        e2 = tensors.Rot(e2)

        rad_mis = crystallography.misorientation(e1, e2, symmetry).torch()

        if angle_type == "degrees":
            rad_mis = torch.rad2deg(rad_mis)

        return rad_mis.detach().cpu().numpy().astype(np.float64, copy=False)

    return dist_edges

# 3by3 tensor norm distance
def diff_norm_3x3(u: np.ndarray, v: np.ndarray) -> float:
    nye_u = u.reshape((3, 3))
    nye_v = v.reshape((3, 3))

    diff = nye_u - nye_v
    diff_norm = np.linalg.norm(diff, ord='fro')

    return diff_norm

def diff_norm_3x3_batch(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    I = edges[:, 0]
    J = edges[:, 1]

    D = X[I] - X[J]
    diff_norm = np.sqrt(np.sum(D * D, axis=1))
    return diff_norm

class SimilarityMetricLibrary:

    """
    To add in more metric in SimilarityMetricLibrary
    Similarity defines as a distance metric function between two samples.
    Here the smaller the value returned by func, the more similar two samples are.
    
    Each metric is defined as a method that returns a SimilarityMetric object.
    It needs to define the feature columns (columns) and the distance function (func).

    SimilarityMetric structure:
    @dataclass
    class SimilarityMetric:
        name: str
        feature_cols: List[str]   # columns required by this metric
        func: DistanceFunction    # metric(u, v) -> float
        dist_edges: Optional[BatchDistanceFunction] = None # X,edges -> (E,) batch version of func, used for vectorized computations.
    
    Bare minimum requirements for a new metric:
    def new_metric(self) -> SimilarityMetric:
        cols = [...]  # list of required feature columns
        return SimilarityMetric(
            name="new_metric",
            feature_cols=cols,
            func=func,
        )

    outside the class define:
        def func(u: np.ndarray, v: np.ndarray) -> float:
            # compute distance between u and v
            return distance_value 
    
    for func(u: np.ndarray, v: np.ndarray) to be callable by multiprocessors,
    it needs to be defined at the top level of the module (not nested inside another function or class).
    
    Important for cols and func:
    The order of feature_cols defines the index mapping: u[i] corresponds to feature_cols[i]

    if the metric can be efficiently computed in a vectorized manner for all edges at once,
    provide dist_edges(X: np.ndarray, edges: np.ndarray) -> np.ndarray
    """

    def von_mises_stress(
        self,
        cols,
    ) -> SimilarityMetric:

        default = ["sxx", "syy", "szz", "sxy", "syz", "sxz"]

        use_cols = cols if cols is not None else default

        return SimilarityMetric(
            name="von_mises_stress",
            feature_cols=use_cols,
            func=von_mises_stress_distance,
            dist_edges=von_mises_stress_distance_batch,
        )
    
    def misorientation(self, symmetry="432", angle_type="degrees", angle_convention="bunge") -> SimilarityMetric:
        
        cols = ["ori_rodrigues_x", "ori_rodrigues_y", "ori_rodrigues_z"]
        
        return SimilarityMetric(
            name="misorientation",
            feature_cols=cols,
            func=lambda u, v: float("nan"),
            dist_edges=make_misorientation_dist_edges(
                symmetry=symmetry, angle_type=angle_type, angle_convention=angle_convention
            ),
        )

    def nye_tensor_norm(self) -> SimilarityMetric:

        cols = [
            "nye_tensor_11", "nye_tensor_12", "nye_tensor_13",
            "nye_tensor_21", "nye_tensor_22", "nye_tensor_23",
            "nye_tensor_31", "nye_tensor_32", "nye_tensor_33",
        ]

        return SimilarityMetric(
            name="nye_tensor_norm",
            feature_cols=cols,
            func=diff_norm_3x3,
            dist_edges=diff_norm_3x3_batch,
        )