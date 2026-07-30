# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Match grains across two load steps via graph message passing and neighbor selection."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict

import pandas as pd
import torch
import tqdm
from torch_geometric.data import Data

from .orientation_helper import mrp_to_matrix, misorientation_matrix


class GraphGrainMatcher:
    """Match grains between two grain graphs (A and B) and write correspondence results."""

    def __init__(
        self,
        graph_a: Any,
        graph_b: Any,
        output_dir: str = "grain_matching_results",
        output_prefix: str = "out_",
    ) -> None:

        self.graph_a = graph_a
        self.graph_b = graph_b
        self.output_dir = output_dir
        self.output_prefix = output_prefix

    def match_grains(
        self,
        message_passing_function=None,
        neighbor_selection_cost_function=None,
        message_passing_iter: int = 6,
        neighbor_selection_param: dict = None,
    ):
        """Run message passing on both graphs, select matches, write and return results."""
        if neighbor_selection_param is None:
            neighbor_selection_param = {
                "lambda": 0.125,
                "iterations": 100,
                "tolerance": 1e-6,
            }

        if message_passing_function is None:
            message_passing_function = (
                GraphGrainMatcher.default_message_passing_function
            )
            use_default = True
            print("Using default message passing function.\n")
        else:
            use_default = False

        if neighbor_selection_cost_function is None:
            neighbor_selection_cost_function = (
                GraphGrainMatcher.default_neighbor_selection_cost_function
            )
            print("Using default neighbor selection cost function.\n")

        print("\nStart grain matching\n")

        print("--- Message Passing on Graph A ---\n")
        Fa, ctx_a = self.message_passing(
            self.graph_a,
            message_passing_function,
            message_passing_iter=message_passing_iter,
            use_default=use_default,
        )

        print("\n--- Message Passing on Graph B ---\n")
        Fb, ctx_b = self.message_passing(
            self.graph_b,
            message_passing_function,
            message_passing_iter=message_passing_iter,
            use_default=use_default,
        )

        print("\n--- Perform neighborhood selection ---\n")
        match_out = self.neighbor_selection(
            Fa,
            Fb,
            neighbor_selection_cost_function=neighbor_selection_cost_function,
            neighbor_selection_param=neighbor_selection_param,
        )

        result = {
            "graph_a_features": Fa,
            "graph_b_features": Fb,
            "graph_a_ctx": ctx_a,
            "graph_b_ctx": ctx_b,
            "match": match_out,
        }

        self.write_results(result)
        return result

    def message_passing(
        self,
        graph: Data,
        message_passing_function,
        message_passing_iter: int = 6,
        use_default: bool = False,
    ):
        """Propagate node features over graph edges for a fixed number of iterations."""
        if not isinstance(graph, Data):
            raise TypeError("graph must be a torch_geometric.data.Data")
        if getattr(graph, "x", None) is None or graph.x.ndim != 2:
            raise ValueError("graph.x must exist and be 2D [N, F]")
        if getattr(graph, "edge_index", None) is None:
            raise ValueError("graph.edge_index missing")

        edge_index = graph.edge_index
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("graph.edge_index must have shape [2, E]")
        if not hasattr(graph, "feature_slices"):
            raise ValueError("graph.feature_slices missing")

        x = graph.x
        device = x.device
        N = x.shape[0]

        if edge_index.device != device or edge_index.dtype != torch.long:
            edge_index = edge_index.to(device=device, dtype=torch.long)

        src0 = edge_index[0]
        dst0 = edge_index[1]

        src = torch.cat([src0, dst0], dim=0)
        dst = torch.cat([dst0, src0], dim=0)
        E = src.shape[0]

        spec, phi_operator = message_passing_function()
        if not isinstance(spec, dict):
            raise TypeError(
                "message_passing_function() must return (spec: dict, phi_operator: callable)"
            )

        required = spec.get("required_node_features", None)
        if not required:
            raise ValueError("spec['required_node_features'] must be a non-empty list")

        graph_slices = graph.feature_slices

        if use_default:
            for name in ("Eul0", "Eul1", "Eul2"):
                if name not in graph_slices:
                    raise KeyError(
                        f"Required feature '{name}' not found in graph.feature_slices"
                    )

        e0a, e0b = graph_slices["Eul0"]
        e1a, e1b = graph_slices["Eul1"]
        e2a, e2b = graph_slices["Eul2"]
        euler = torch.cat([x[:, e0a:e0b], x[:, e1a:e1b], x[:, e2a:e2b]], dim=1)
        if euler.ndim != 2 or euler.shape[1] != 3:
            raise ValueError(f"Euler must be [N,3], got {euler.shape}")

        blocks = []
        F_slices = {}
        start = 0

        if use_default:
            for name in ("X", "Y", "Z"):
                if name not in graph_slices:
                    raise KeyError(
                        f"Required feature '{name}' not found in graph.feature_slices"
                    )
                a, b = graph_slices[name]
                width = b - a
                blocks.append(x[:, a:b])
                F_slices[name] = (start, start + width)
                start += width

            blocks.append(torch.zeros((N, 1), device=device, dtype=x.dtype))
            F_slices["M"] = (start, start + 1)
            start += 1

        else:
            for name in required:
                if name not in graph_slices:
                    raise KeyError(
                        f"Required feature '{name}' not found in graph.feature_slices"
                    )
                a, b = graph_slices[name]
                width = b - a
                blocks.append(x[:, a:b])
                F_slices[name] = (start, start + width)
                start += width

        d = start
        F = torch.cat(blocks, dim=1)

        ctx = {
            "F_slices": F_slices,
            "required_node_features": list(required),
            "euler": euler,
            "edge_src": src,
            "edge_dst": dst,
        }

        message_passing_iter = int(message_passing_iter)
        if message_passing_iter < 0:
            raise ValueError("message_passing_iter must be >= 0")

        for k in tqdm.tqdm(range(message_passing_iter), desc="Message Passing"):
            F_src = F[src]
            F_dst = F[dst]

            # phi_operator MUST accept (F_src, F_dst, ctx, k)
            # phi_operator must accept (F_src, F_dst, ctx, k) and return [E, d]
            Phi = phi_operator(F_src, F_dst, ctx, k)
            if (not torch.is_tensor(Phi)) or Phi.ndim != 2 or Phi.shape != (E, d):
                raise ValueError(
                    f"phi_operator must return tensor of shape [E, {d}], got {getattr(Phi, 'shape', None)}"
                )

            m = torch.zeros((N, d), dtype=F.dtype, device=F.device)
            m.index_add_(0, dst, Phi)
            F = F + m

        return F, ctx

    def neighbor_selection(
        self,
        graph_a_features,
        graph_b_features,
        neighbor_selection_cost_function,
        neighbor_selection_param=None,
    ):
        """Assign each grain in A to a grain in B by iterative cost-minimizing selection."""
        if neighbor_selection_param is None:
            neighbor_selection_param = {
                "lambda": 0.125,
                "iterations": 100,
                "tolerance": 1e-6,
                "topk": 25,
                "chunk": 1000,
            }

        lam = float(neighbor_selection_param.get("lambda", 0.125))
        max_it = int(neighbor_selection_param.get("iterations", 100))
        tol = float(neighbor_selection_param.get("tolerance", 1e-6))
        topk = int(neighbor_selection_param.get("topk", 25))
        chunk = int(neighbor_selection_param.get("chunk", 1024))

        Fa = graph_a_features
        Fb = graph_b_features

        if not torch.is_tensor(Fa) or not torch.is_tensor(Fb):
            raise TypeError("graph_*_features must be tensors.")

        device = Fa.device
        Fb = Fb.to(device=device, dtype=Fa.dtype)

        Na, d = Fa.shape
        Nb, d2 = Fb.shape
        if d != d2:
            raise ValueError(f"Feature dims mismatch: {d} vs {d2}")

        def build_neighbors(graph):
            ei = graph.edge_index
            if ei.device != device:
                ei = ei.to(device)
            src = torch.cat([ei[0], ei[1]], dim=0)
            dst = torch.cat([ei[1], ei[0]], dim=0)

            src_cpu = src.detach().cpu()
            dst_cpu = dst.detach().cpu()

            N = int(
                getattr(graph, "num_nodes", 0)
                or (max(src_cpu.max().item(), dst_cpu.max().item()) + 1)
            )
            neigh = [set() for _ in range(N)]
            for s, t in zip(src_cpu.tolist(), dst_cpu.tolist()):
                if s != t:
                    neigh[s].add(t)
            return neigh

        neigh_a = build_neighbors(self.graph_a)
        neigh_b = build_neighbors(self.graph_b)

        cand_j = torch.empty((Na, topk), dtype=torch.long, device=device)

        for i0 in range(0, Na, chunk):
            i1 = min(Na, i0 + chunk)
            Fi = Fa[i0:i1]
            dist = torch.cdist(Fi, Fb, p=2.0) ** 2
            _, idx = torch.topk(dist, k=min(topk, Nb), dim=1, largest=False)
            cand_j[i0:i1, : idx.shape[1]] = idx
            if idx.shape[1] < topk:
                cand_j[i0:i1, idx.shape[1] :] = -1

        a_to_b = [-1] * Na
        mean_cost_history = []

        _it = max_it - 1  # defined even if max_it == 0 (loop body never runs)
        for _it in tqdm.tqdm(range(max_it), desc="Neighbor Selection"):

            best_for_i = [(-1, math.inf)] * Na  # (j, cost)
            for i in range(Na):
                for j in cand_j[i].tolist():
                    if j < 0:
                        continue
                    c = neighbor_selection_cost_function(
                        Fa, Fb, neigh_a, neigh_b, a_to_b, i, j, lam
                    )
                    if c < best_for_i[i][1]:
                        best_for_i[i] = (j, c)

            claims = {}
            for i, (j, c) in enumerate(best_for_i):
                if j < 0 or not math.isfinite(c):
                    continue
                if (j not in claims) or (c < claims[j][1]):
                    claims[j] = (i, c)

            new_a_to_b = [-1] * Na
            for j, (i, c) in claims.items():
                new_a_to_b[i] = j

            assigned = [(i, j) for i, j in enumerate(new_a_to_b) if j != -1]
            if not assigned:
                break

            costs = [
                neighbor_selection_cost_function(
                    Fa, Fb, neigh_a, neigh_b, new_a_to_b, i, j, lam
                )
                for i, j in assigned
            ]
            mean_cost = float(sum(costs) / len(costs))
            mean_cost_history.append(mean_cost)

            a_to_b = new_a_to_b

        if _it < max_it - 1:
            print("Converged before max iterations.")

        a_to_b_tensor = torch.tensor(a_to_b, dtype=torch.long, device=device)

        matched_i = torch.nonzero(a_to_b_tensor >= 0, as_tuple=False).view(-1)
        matched_j = a_to_b_tensor[matched_i]
        matches = torch.stack([matched_i, matched_j], dim=1)

        a_to_b_cpu = a_to_b_tensor.detach().cpu().tolist()

        final_costs = torch.empty((matches.shape[0],), dtype=Fa.dtype, device=device)
        for k in range(matches.shape[0]):
            i = int(matches[k, 0].item())
            j = int(matches[k, 1].item())
            final_costs[k] = neighbor_selection_cost_function(
                Fa, Fb, neigh_a, neigh_b, a_to_b_cpu, i, j, lam
            )

        return {
            "matches": matches,
            "costs": final_costs,
            "a_to_b": a_to_b_tensor,
            "mean_cost_history": mean_cost_history,
            "params": {
                "lambda": lam,
                "iterations": max_it,
                "tolerance": tol,
                "topk": topk,
                "chunk": chunk,
            },
        }

    def write_results(self, result: Dict[str, Any]) -> None:
        """Write matches CSV, per-node A->B mapping, embeddings (.pt), and meta JSON."""
        out_dir = getattr(self, "output_dir", "match_results")
        run = getattr(self, "run_name", "run")

        os.makedirs(out_dir, exist_ok=True)

        Fa = result["graph_a_features"]
        Fb = result["graph_b_features"]
        match_out = result["match"]

        matches = match_out["matches"].detach().cpu()
        costs = match_out["costs"].detach().cpu()
        a_to_b = match_out["a_to_b"].detach().cpu()

        df_matches = pd.DataFrame(
            {
                "i_in_A": matches[:, 0].numpy() if matches.numel() else [],
                "j_in_B": matches[:, 1].numpy() if matches.numel() else [],
                "cost": costs.numpy() if costs.numel() else [],
            }
        )
        df_matches.to_csv(os.path.join(out_dir, f"{run}_matches.csv"), index=False)

        df_map = pd.DataFrame(
            {"i_in_A": list(range(a_to_b.numel())), "j_in_B": a_to_b.numpy()}
        )
        df_map.to_csv(os.path.join(out_dir, f"{run}_a_to_b.csv"), index=False)

        torch.save(Fa.detach().cpu(), os.path.join(out_dir, f"{run}_Fa.pt"))
        torch.save(Fb.detach().cpu(), os.path.join(out_dir, f"{run}_Fb.pt"))

        # metadata (JSON-serializable only)
        meta = {
            "neighbor_selection_params": match_out.get("params", {}),
            "mean_cost_history": match_out.get("mean_cost_history", []),
            "graph_a_feature_names": getattr(self.graph_a, "feature_names", None),
            "graph_b_feature_names": getattr(self.graph_b, "feature_names", None),
            "message_passing_required_features": result["graph_a_ctx"].get(
                "required_node_features", None
            ),
        }
        with open(
            os.path.join(out_dir, f"{run}_meta.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def default_message_passing_function(
        angle_convention="kocks",  # pylint: disable=unused-argument  # kept for interface parity
        angle_type="radians",
        symmetry="432",
    ):
        """Return (spec, phi_operator) for the default misorientation message-passing scheme."""
        spec = {
            "required_node_features": ["X", "Y", "Z", "Eul0", "Eul1", "Eul2"],
        }

        def phi_operator(F_src, F_dst, ctx, k: int):
            device = F_src.device
            dtype = F_src.dtype
            E = F_src.shape[0]

            Fs = ctx["F_slices"]
            if "M" not in Fs:
                raise ValueError(
                    "default_message_passing_function requires packed 'M' in ctx['F_slices']."
                )

            aM, bM = Fs["M"]

            dX = torch.zeros((E, 1), device=device, dtype=dtype)
            dY = torch.zeros((E, 1), device=device, dtype=dtype)
            dZ = torch.zeros((E, 1), device=device, dtype=dtype)

            if k == 0:
                euler = ctx["euler"]
                src = ctx["edge_src"]
                dst = ctx["edge_dst"]
                e1 = euler[src]
                e2 = euler[dst]

                # Assumes graintrace MRP (Gibbs) params; for Euler input swap in
                # euler_to_matrix(e, angle_convention, angle_type) below.
                R1 = mrp_to_matrix(e1)
                R2 = mrp_to_matrix(e2)

                rad_mis = misorientation_matrix(R1, R2, symmetry, angle_type="radians")
                if rad_mis.ndim == 1:
                    rad_mis = rad_mis.unsqueeze(1)
                elif not (rad_mis.ndim == 2 and rad_mis.shape[1] == 1):
                    raise ValueError(
                        f"Expected misorientation shape [E] or [E,1], got {rad_mis.shape}"
                    )

                mis = torch.rad2deg(rad_mis) if angle_type == "degrees" else rad_mis
                Phi = torch.zeros((E, F_src.shape[1]), device=device, dtype=dtype)
                Phi[:, 0:1] = dX
                Phi[:, 1:2] = dY
                Phi[:, 2:3] = dZ
                Phi[:, aM:bM] = mis
                return Phi

            l2 = torch.linalg.norm(F_dst - F_src, dim=1, ord=2).unsqueeze(1)

            Phi = torch.zeros((E, F_src.shape[1]), device=device, dtype=dtype)
            Phi[:, 0:1] = dX
            Phi[:, 1:2] = dY
            Phi[:, 2:3] = dZ
            Phi[:, aM:bM] = l2
            return Phi

        return spec, phi_operator

    @staticmethod
    def default_neighbor_selection_cost_function(
        Fa, Fb, neigh_a, neigh_b, a_to_b, i, j, lam
    ):
        """Default matching cost: feature distance plus a neighbor-consistency term."""

        def psi(u, v):
            return -torch.sum((u - v) ** 2).item()

        # base term: ||Fi - Fj||^2
        cost = -psi(Fa[i], Fb[j])

        ni = neigh_a[i]
        nj = neigh_b[j]
        if not ni or not nj:
            return cost

        cons = 0.0
        cnt = 0
        for p in ni:
            q = a_to_b[p]
            if q is None or q < 0:
                continue
            if q in nj:
                cons += psi(Fa[p], Fb[q])
                cnt += 1

        # degree-normalize so high-degree nodes don't automatically win
        if cnt > 0:
            cost += lam * (cons / cnt)

        return cost
