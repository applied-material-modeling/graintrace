import sys
import os
import json
from pathlib import Path

# Ensure project root in path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder


def main():
    params = json.loads(sys.stdin.read())

    builder = VoronoiMeshBuilder(
        input_csv=params["input_csv"],
        output_dir=params["output_dir"],
        bounding_box=params["bounding_box"],
        dim=params["dim"],
        weighted=params["weighted"],
        auto_fix_bbox=params["auto_fix_bbox"],
        bbox_fix_mode=params["bbox_fix_mode"],
        bbox_tolerance=params["bbox_tolerance"],
        auto_rotate=params["auto_rotate"],
        rotate_angles=params["rotate_angles"],
        rotate_convention=params["rotate_convention"],
        unit=params["unit"],
        angle_identifier=params["angle_identifier"],
        orientation_descriptor=params["orientation_descriptor"],
        orientation_active_convention=params["orientation_active_convention"],
        elastic_strain_identifier=params["elastic_strain_identifier"],
        strain_unit=params["strain_unit"],
    )

    builder.build_voronoi(
        generate_mesh=params["generate_mesh"],
        mesh_quality_min=params["mesh_quality_min"],
        relative_el_size=params["relative_el_size"],
        option=params["option"],
        CVT_iter=params["CVT_iter"],
        morphoalgo=params["morphoalgo"],
    )

    print("Voronoi successfully built.")


if __name__ == "__main__":
    main()