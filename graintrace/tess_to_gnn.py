from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
from torch_geometric.data import Data
import os


class NeperTessToGraphNN:
    """
    Parse a Neper .tess file and build a graph representation (pytorch_geometric Data)
    with grains as nodes and faces as edges.
    """

    def __init__(
        self,
        tess_path,
        geometry_cell_file=None,
        geometry_face_file=None,
        device="cpu",
        dtype=torch.float64,
    ):

        self.tess_path = tess_path
        self.device = device
        self.dtype = dtype

        # parsed geometry
        self.cell_seeds = None
        self.orientations = None
        self.vertices = None
        self.edges = None
        self.face_vertices = []
        self.face_edges = []
        self.cell_to_faces = []
        self.tessfile = tess_path

        # metadata
        self.ori_type = None

        # registries
        self.node_feature_registry = {}
        self.edge_feature_registry = {}
        self.active_node_features = []
        self.active_edge_features = []

        # parse file
        self.parse_tess()

        # Geometry placeholders
        self.cell_centroid = []
        self.cell_vol = []
        self.face_id = []
        self.face_centroid = []
        self.face_area = []

        # If geometry files exist, load them; else compute with Neper
        if geometry_cell_file or geometry_face_file:
            self.load_geometry_information(geometry_cell_file, geometry_face_file)
        else:
            self.run_neper_for_geometry_information(self.tess_path)

        # call default feature registration
        self.register_default_features()
        self.activate_all_registered_features()

    # install NEPER to easily access .tess files geometry information
    def install_neper(self):
        pass

    def load_geometry_information(self, cell_file, face_file):
        pass

    def run_neper_for_geometry_information(self, tess_path):
        pass

    # ---------- PARSING ----------
    def parse_tess(self):
        """
        Parse the Neper .tess file.
        """
        if not os.path.exists(self.tess_path):
            raise FileNotFoundError(f"Tessellation file not found: {self.tess_path}")

        # Fresh storage per call — no shared state
        sections = {
            "cell": {
                "seed": [],
                "ori": [],
            },
            "vertex": [],
            "edge": [],
            "face": [],
            "polyhedron": [],
        }

        with open(self.tess_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        section = None
        subsection = None

        for line in lines:
            if line.startswith("**"):
                section = line.strip("*").lower()
                subsection = None
                continue

            if section == "cell":
                if line.startswith("*"):
                    name = line.strip("*").lower()
                    subsection = name if name in sections["cell"] else None
                    continue

                if subsection:
                    sections["cell"][subsection].append(line)
                continue

            if section in sections:
                sections[section].append(line)

        self._parse_sections(sections)
        return self

    def _parse_sections(self, sections):
        """
        Extract essential geometric/topological data from Neper .tess file.
        For edges, faces_vertices, edges, and polyhdera (cells), there are no associated IDS
        IDs are implicit based on order (1-based in file, 0-based in code).
        1. Cells: seeds (x,y,z,w), orientations (type, values)
        2. Vertices: (x,y,z)
        3. Edges: (v1, v2)
        4. Faces: (nverts, v1, v2, ...), (nedges, e1, e2, ...)
        5. Polyhedra: (nfaces, f1, f2, ...)
        Note: face_to_cells can be constructed from polyhedra data
        - signed edges are kept for potential orientation use
        """
        import re

        # =============== 1. CELLS ===============
        cell_data = sections.get("cell", {})
        self.cell_seeds = torch.zeros((0, 4), dtype=self.dtype, device=self.device)
        # to be fixed , sometimes orientations provide 3 or 9
        self.orientations = torch.zeros((0, 3), dtype=self.dtype, device=self.device)
        self.ori_type = "none"

        # *seed
        if "seed" in cell_data:
            lines = [ln.strip() for ln in cell_data["seed"] if ln.strip()]
            if len(lines) > 0 and re.match(r"^\d+$", lines[0]):  # skip count
                lines = lines[1:]
            seeds = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    _, x, y, z, w = parts[:5]
                    seeds.append([float(x), float(y), float(z), float(w)])
            if seeds:
                self.cell_seeds = torch.tensor(
                    seeds, dtype=self.dtype, device=self.device
                )

        # *ori
        if "ori" in cell_data:
            lines = [ln.strip() for ln in cell_data["ori"] if ln.strip()]
            if lines:
                self.ori_type = lines[0].split(":")[0].lower()
                data_lines = lines[1:]
                ori_vals = []
                for line in data_lines:
                    if re.match(r"^[-\d]", line):
                        # fix here if 3 or 9 values
                        vals = list(map(float, line.split()))[:3]
                        ori_vals.append(vals)
                if ori_vals:
                    self.orientations = torch.tensor(
                        ori_vals, dtype=self.dtype, device=self.device
                    )

        # =============== 2. VERTICES ===============
        if "vertex" in sections:
            lines = [ln.strip() for ln in sections["vertex"] if ln.strip()]
            if len(lines) > 0 and re.match(r"^\d+$", lines[0]):  # skip count
                lines = lines[1:]
            verts = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 4:
                    _, x, y, z = parts[:4]
                    verts.append([float(x), float(y), float(z)])
            self.vertices = torch.tensor(verts, dtype=self.dtype, device=self.device)

        # =============== 3. EDGES ===============
        if "edge" in sections:
            lines = [ln.strip() for ln in sections["edge"] if ln.strip()]
            if len(lines) > 0 and re.match(r"^\d+$", lines[0]):  # skip count
                lines = lines[1:]
            edges = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    _, v1, v2 = parts[:3]
                    edges.append([int(v1) - 1, int(v2) - 1])
            self.edges = torch.tensor(edges, dtype=torch.long, device=self.device)

        # =============== 4. FACES (keep signed edges) ===============
        if "face" in sections:
            lines = [ln.strip() for ln in sections["face"] if ln.strip()]
            if len(lines) > 0 and re.match(r"^\d+$", lines[0]):  # skip count
                lines = lines[1:]

            face_vertices = []
            face_edges = []
            i = 0
            while i + 3 < len(lines):
                if re.match(r"^\d+\s", lines[i]):
                    # line 1
                    header = list(map(int, lines[i].split()))
                    _, nverts = header[0:2]
                    verts = [v - 1 for v in header[2 : 2 + nverts]]

                    # line 2
                    edge_line = list(map(int, lines[i + 1].split()))
                    nedges = edge_line[0]
                    edges_signed = [
                        e - 1 if e > 0 else -(abs(e) - 1)
                        for e in edge_line[1 : 1 + nedges]
                    ]

                    face_vertices.append(verts)
                    face_edges.append(edges_signed)
                    i += 4
                else:
                    i += 1

            self.face_vertices = face_vertices
            self.face_edges = face_edges

        # =============== 5. POLYHEDRA ===============
        if "polyhedron" in sections:
            lines = [ln.strip() for ln in sections["polyhedron"] if ln.strip()]
            if len(lines) > 0 and re.match(r"^\d+$", lines[0]):  # skip count
                lines = lines[1:]

            cell_to_faces = []
            for line in lines:
                parts = list(map(int, line.split()))
                if len(parts) >= 2:
                    pid, nfaces = parts[:2]

                    # keep sign but fix 1-based indexing
                    faces_signed = [
                        f - 1 if f > 0 else -(abs(f) - 1) for f in parts[2 : 2 + nfaces]
                    ]

                    cell_to_faces.append(faces_signed)

            self.cell_to_faces = cell_to_faces

    def validate_topology(self, verbose: bool = True) -> bool:
        """
        Validate the graph connectivity:
        ensure each face belongs to 1 or 2 cells.
        """
        num_faces = len(self.face_vertices)
        face_to_cells = {}

        for cell_id, faces in enumerate(self.cell_to_faces):
            for f in faces:
                fid = abs(f)
                if fid >= num_faces:
                    continue
                face_to_cells.setdefault(fid, []).append(cell_id)

        # classify faces
        isolated = [f for f, cells in face_to_cells.items() if len(cells) == 0]
        internal = [f for f, cells in face_to_cells.items() if len(cells) == 2]
        boundary = [f for f, cells in face_to_cells.items() if len(cells) == 1]
        nonmanifold = [f for f, cells in face_to_cells.items() if len(cells) > 2]

        if verbose:
            print("=== Face Connectivity Check ===")
            print(f"Total faces: {num_faces}")
            print(f"Internal (2 cells): {len(internal)}")
            print(f"Boundary (1 cell): {len(boundary)}")
            print(f"Non-manifold (>2 cells): {len(nonmanifold)}")
            print(f"Isolated (0 cell): {len(isolated)}")

            if len(nonmanifold) or len(isolated):
                raise ValueError(
                    "Faces not properly shared. "
                    "Either non-manifold or isolated faces detected."
                )
            else:
                print("\nAll faces belong to 1 or 2 cells.\n")

        return {
            "internal": internal,
            "boundary": boundary,
            "nonmanifold": nonmanifold,
            "isolated": isolated,
        }

    # ---------- FEATURE REGISTRATION ----------
    # ---------- ADD IN AS NEEDED --------------
    def register_default_features(self):
        """
        Add more features as needed
        """

        # --- Cell centroid ---
        def seed_centroid(self):
            # right now return just cell_seeds , this will be implemented later
            return self.cell_seeds
            # return self.cell_centroid

        # these are the information required later on for Gary Approximation
        # will be implemented later
        # --- Cell volume ---
        # def node_volume(self):
        #     return self.cell_vol

        # --- Face centroid / area for edge-level ---
        # def edge_centroid(self, edge_index):
        #     return self.face_centroid[:edge_index.shape[1]]

        # def edge_area(self, edge_index):
        #     return self.face_area[:edge_index.shape[1]]

        self.node_feature_registry["seed_centroid"] = seed_centroid
        # self.node_feature_registry["volume"] = node_volume
        # self.edge_feature_registry["centroid"] = edge_centroid
        # self.edge_feature_registry["area"] = edge_area

    def register_dataframe_features(self, data: Any, verbose: bool = True) -> None:
        """
        Register every column in a pandas DataFrame as a node feature.
        """

        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError("register_dataframe_features expects a pandas DataFrame.")

        n_cells = int(self.cell_seeds.shape[0])
        n_df = int(len(data))
        if n_df != n_cells:
            raise ValueError(
                f"DataFrame length mismatch: len(data)={n_df} but num_cells={n_cells}. "
                "Row order/alignment is required."
            )

        # storage for tensors created from dataframe columns
        if not hasattr(self, "_df_node_tensors") or self._df_node_tensors is None:
            self._df_node_tensors = {}

        if verbose:
            print("this dataframe has the following columns:")
            print(list(data.columns))

        for col in data.columns:
            s = data[col]

            if pd.api.types.is_bool_dtype(s):
                s_num = s.astype(float)
            elif pd.api.types.is_numeric_dtype(s):
                s_num = s.astype(float)
            else:
                s_num = pd.to_numeric(s, errors="coerce").astype(float)

            arr = s_num.to_numpy(dtype=float, copy=False)
            t = torch.tensor(arr, dtype=self.dtype, device=self.device).view(n_cells, 1)

            self._df_node_tensors[col] = t

            def _feat(self_ref, name=col):
                return self_ref._df_node_tensors[name]

            self.node_feature_registry[col] = _feat

        # Ensure new features are active
        self.activate_all_registered_features()

    def activate_all_registered_features(self) -> None:
        """Activate all currently registered node and edge features."""
        self.active_node_features = list(self.node_feature_registry.keys())
        self.active_edge_features = list(self.edge_feature_registry.keys())

    # ---------- BUILD GRAPH ----------
    def build_cell_graph(self):
        """Construct a torch_geometric Data graph using selected features."""

        if not self.cell_to_faces:
            raise RuntimeError("Cell-to-face topology missing — parse_tess() first.")

        # Make sure the topology is valid before building the graph
        self.validate_topology()

        # --- Build edge_index from shared faces (node connectivity) ---
        face_to_cells = {}
        for cell_id, faces in enumerate(self.cell_to_faces):
            for f in faces:
                face_to_cells.setdefault(abs(f), []).append(cell_id)
        edges = [cells for f, cells in face_to_cells.items() if len(cells) == 2]
        if not edges:
            raise ValueError("No shared faces found; check tessellation integrity.")

        # edge_index shape should be [2, num_edges] where position 0 is source and 1 is target
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).T

        # --- Node features ---
        node_feats = []
        feature_slices = {}

        start = 0
        for name in self.active_node_features:
            func = self.node_feature_registry.get(name)
            if func is None:
                raise KeyError(f"Unregistered node feature: {name}")

            feat = func(self)
            if feat.ndim != 2:
                raise ValueError(
                    f"Node feature '{name}' must be 2D, got shape {feat.shape}"
                )

            k = feat.shape[1]
            feature_slices[name] = (start, start + k)
            start += k

            node_feats.append(feat)

        # x - node features
        x = (
            torch.cat(node_feats, dim=1)
            if node_feats
            else torch.empty(
                (len(self.cell_seeds), 0), dtype=self.dtype, device=self.device
            )
        )

        # --- Edge features ---
        edge_feats = []
        for name in self.active_edge_features:
            func = self.edge_feature_registry.get(name)
            if func is None:
                raise KeyError(f"Unregistered edge feature: {name}")
            edge_feats.append(func(self, edge_index))

        # edge_attr - edge features, shape should be [num_edges, num_edge_features]
        edge_attr = (
            torch.cat(edge_feats, dim=1)
            if edge_feats
            else torch.empty(
                (edge_index.shape[1], 0), dtype=self.dtype, device=self.device
            )
        )

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        graph.feature_names = list(self.active_node_features)
        graph.feature_slices = feature_slices
        graph.edge_feature_names = list(self.active_edge_features)

        return graph

    def visualize_graph_2D(
        self,
        graph,
        node_attr=None,
        edge_attr=None,
        cmap="viridis",
        outpath="graph_2D.png",
        show_node_labels=True,
        show_edge_labels=True,
    ):

        import matplotlib.pyplot as plt
        import networkx as nx

        """
        Visualize the cell-cell graph using NetworkX.
        Colors nodes and/or edges based on the given attributes. If attributes have more than 1 dimension,
        their norms are used for coloring.

        Args:
            graph (torch_geometric.data.Data): The graph object (from build_cell_graph()).
            node_attr (str or torch.Tensor, optional): Node attribute name (if registered)
                or tensor of shape [num_nodes] or [num_nodes, k].
            edge_attr (str or torch.Tensor, optional): Edge attribute name (if registered)
                or tensor of shape [num_edges] or [num_edges, k].
            cmap (str): Matplotlib colormap name.
        """

        G = nx.Graph()
        edge_index = graph.edge_index.cpu().numpy().T
        G.add_edges_from(edge_index)

        num_nodes = graph.x.shape[0]
        pos = nx.spring_layout(G, seed=42)  # deterministic layout

        # --- Node coloring ---
        if isinstance(node_attr, str):
            if node_attr not in self.node_feature_registry:
                raise KeyError(f"Node attribute '{node_attr}' not registered.")
            vals = self.node_feature_registry[node_attr](self)
        elif isinstance(node_attr, torch.Tensor):
            vals = node_attr
        else:
            vals = torch.zeros((num_nodes, 1))

        vals = vals.detach().cpu()
        if vals.ndim > 1:
            vals = vals.norm(dim=1)
        node_colors = vals.squeeze().numpy()

        # --- Edge coloring ---
        if edge_attr is not None:
            num_edges = graph.edge_index.shape[1]
            if isinstance(edge_attr, str):
                if edge_attr not in self.edge_feature_registry:
                    raise KeyError(f"Edge attribute '{edge_attr}' not registered.")
                e_vals = self.edge_feature_registry[edge_attr](self, graph.edge_index)
            elif isinstance(edge_attr, torch.Tensor):
                e_vals = edge_attr
            else:
                e_vals = torch.zeros((num_edges, 1))
            e_vals = e_vals.detach().cpu()
            if e_vals.ndim > 1:
                e_vals = e_vals.norm(dim=1)
            edge_colors = e_vals.squeeze().numpy()
        else:
            edge_colors = "black"

        # --- Draw ---
        nx.draw(
            G,
            pos,
            node_color=node_colors,
            edge_color=edge_colors,
            cmap=cmap,
            node_size=200,
            with_labels=False,
            font_size=8,
        )

        # --- Add node labels ---
        if show_node_labels:
            node_labels = {i: str(i) for i in range(num_nodes)}
            nx.draw_networkx_labels(
                G, pos, labels=node_labels, font_size=8, font_color="white"
            )

        # --- Add edge labels ---
        if show_edge_labels:
            edge_labels = {edge: str(i) for i, edge in enumerate(G.edges())}
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_size=6, rotate=False
            )

        plt.savefig(outpath, dpi=300)
        plt.close()

    def visualize_graph_3D(
        self,
        graph,
        outpath="graph_3D.png",
        node_attr=None,
        edge_attr=None,
        cmap="viridis",
        show_node_labels=False,
        show_edge_labels=False,
    ):
        """
        3D visualization of the cell-cell graph using centroid positions if available.
        Falls back to seed positions if centroids are not yet computed.
        """

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        if "centroid" in self.node_feature_registry:
            pos = self.node_feature_registry["centroid"](self).detach().cpu()
        else:
            pos = self.cell_seeds[:, :3].detach().cpu()
            print(
                "WARNING: Using seed positions (centroid node feature not available)."
            )

        num_nodes = pos.shape[0]
        edges = graph.edge_index.t().cpu()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # --- Node color ---
        if isinstance(node_attr, str):
            vals = self.node_feature_registry[node_attr](self)
        elif isinstance(node_attr, torch.Tensor):
            vals = node_attr
        else:
            vals = torch.zeros((num_nodes, 1), dtype=self.dtype)
        vals = vals.detach().cpu()
        if vals.ndim > 1:
            vals = vals.norm(dim=1)
        node_colors = vals.squeeze().numpy()

        # --- Edge color ---
        if edge_attr is not None:
            e_vals = edge_attr
            if e_vals.ndim > 1:
                e_vals = e_vals.norm(dim=1)
            edge_colors = e_vals.detach().cpu().squeeze().numpy()
        else:
            edge_colors = "black"

        # --- Draw edges ---
        for i, (src, dst) in enumerate(edges):
            x = [pos[src, 0], pos[dst, 0]]
            y = [pos[src, 1], pos[dst, 1]]
            z = [pos[src, 2], pos[dst, 2]]
            color = (
                edge_colors[i]
                if isinstance(edge_colors, (list, torch.Tensor))
                else "black"
            )
            ax.plot(x, y, z, color=color, alpha=0.7, linewidth=0.8)

        # --- Draw nodes ---
        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=node_colors,
            cmap=cmap,
            s=40,
            edgecolor="k",
        )

        if show_node_labels:
            for i, (x, y, z) in enumerate(pos):
                ax.text(x, y, z, str(i), fontsize=6, color="black")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()
        plt.savefig(outpath, dpi=300)


if __name__ == "__main__":
    tess_path = "output_test/voronoi.tess"

    parser = NeperTessToGraphNN(tess_path=tess_path, device="cpu", dtype=torch.float64)

    print("\n=== Basic Stats ===")
    print("num_cells    :", parser.cell_seeds.shape[0])
    print("num_vertices :", parser.vertices.shape[0])
    print("num_edges    :", parser.edges.shape[0])
    print("num_faces    :", len(parser.face_vertices))
    print("ori_type     :", parser.ori_type)

    print("\n=== Quick Data Check ===")
    print("cell_seeds:\n", parser.cell_seeds[:5])  # first 5 cells
    print("orientations:\n", parser.orientations[:5])  # first 5 orientations
    print("vertices:\n", parser.vertices[:5])  # first 5 vertices
    print("edges:\n", parser.edges[:5])  # first 5 edges
    print("face_vertices (first 5):")
    for fv in parser.face_vertices[:5]:
        print(" ", fv)
    print("face_edges (first 5):")
    for fe in parser.face_edges[:5]:
        print(" ", fe)
    print("polyhedra (first 5 cells → their faces):")
    for faces in parser.cell_to_faces[:5]:
        print(" ", faces)

    print("\n=== Graph information ===")
    graph = parser.build_cell_graph()
    parser.visualize_graph_2D(graph)
    parser.visualize_graph_3D(graph)
