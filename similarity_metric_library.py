import numpy as np
from cluster_indicator import SimilarityMetric  # adjust import path

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
    
    Bare minimum requirements for a new metric:
    def new_metric(self) -> SimilarityMetric:
        cols = [...]  # list of required feature columns
        def func(u: np.ndarray, v: np.ndarray) -> float:
            # compute distance between u and v
            return distance_value 
        return SimilarityMetric(
            name="new_metric",
            feature_cols=cols,
            func=func,
        )

    Important for cols and func:
    The order of feature_cols defines the index mapping: u[i] corresponds to feature_cols[i]
    """

    def von_mises_stress(self) -> SimilarityMetric:

        cols = ["sxx", "syy", "szz", "sxy", "syz", "sxz"]

        def func(u: np.ndarray, v: np.ndarray) -> float:
            sxx_u, syy_u, szz_u, sxy_u, syz_u, sxz_u = u
            sxx_v, syy_v, szz_v, sxy_v, syz_v, sxz_v = v

            def von_mises(sxx, syy, szz, sxy, syz, sxz):
                term1 = ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) / 2
                term2 = 3 * (sxy ** 2 + syz ** 2 + sxz ** 2)
                return np.sqrt(term1 + term2)
            vm_u = von_mises(sxx_u, syy_u, szz_u, sxy_u, syz_u, sxz_u)
            vm_v = von_mises(sxx_v, syy_v, szz_v, sxy_v, syz_v, sxz_v)

            norm_abs_diff = abs(vm_u - vm_v) / (abs(vm_u) + abs(vm_v) + 1e-8)

            return norm_abs_diff

        return SimilarityMetric(
            name="von_mises_stress",
            feature_cols=cols,
            func=func,
        )

    def misorientation(self, symmetry='432',angle_type="degrees") -> SimilarityMetric:

        cols = ["ori_rodrigues_x", "ori_rodrigues_y", "ori_rodrigues_z"]

        def func(u: np.ndarray, v: np.ndarray) -> float:
            import neml2
            import torch
            from neml2 import tensors
            from neml2 import crystallography
            
            e1 = torch.tensor(u, dtype=torch.float64)
            e2 = torch.tensor(v, dtype=torch.float64)
            if e1.ndim == 1:
                e1 = e1.unsqueeze(0)
            if e2.ndim == 1:
                e2 = e2.unsqueeze(0)

            e1 = tensors.Rot(e1)
            e2 = tensors.Rot(e2)

            rad_mis = crystallography.misorientation(e1, e2, symmetry).torch()

            if angle_type == "degrees":
                return torch.rad2deg(rad_mis)

            return rad_mis

        return SimilarityMetric(
            name="misorientation",
            feature_cols=cols,
            func=func,
        )

    def nye_tensor_norm(self) -> SimilarityMetric:

        cols = [
            "nye_tensor_11", "nye_tensor_12", "nye_tensor_13",
            "nye_tensor_21", "nye_tensor_22", "nye_tensor_23",
            "nye_tensor_31", "nye_tensor_32", "nye_tensor_33",
        ]

        def func(u: np.ndarray, v: np.ndarray) -> float:
            nye_u = u.reshape((3, 3))
            nye_v = v.reshape((3, 3))

            diff = nye_u - nye_v
            diff_norm = np.linalg.norm(diff, ord='fro')

            return diff_norm

        return SimilarityMetric(
            name="nye_tensor_norm",
            feature_cols=cols,
            func=func,
        )