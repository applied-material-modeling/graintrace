import math
import os
import json
import signal
import subprocess
from textual.app import ComposeResult
from textual.widgets import Header, Footer, Static, Button, Input, Label
from textual.containers import Vertical, Center, Horizontal, VerticalScroll
from textual.screen import Screen


class VoronoiScreen(Screen):
    """Configuration screen for Voronoi mesh building (non-blocking execution)."""

    current_proc: subprocess.Popen | None = None

    CSS = """
    Screen {
        align: center middle;
    }

    #title {
        text-align: center;
        background: blue 50%;
        border: wide red;
        color: white;
        margin: 2 10;
    }

    #basic-title {
        color: ansi_bright_cyan;
        margin-top: 2;
        margin-bottom: 1;
        text-style: bold;
    }

    #advanced-title {
        color: ansi_bright_magenta;
        margin-top: 3;
        margin-bottom: 1;
        text-style: bold;
    }

    Input {
        width: 100;
        margin: 1 1;
    }

    Label {
        margin: 0 0 0 1;
    }

    .basic-label {
        color: yellow;
    }

    .advanced-label {
        color: white;
    }

    Button {
        content-align: center middle;
        margin: 1 2;
        width: 25;
    }

    Button:hover {
        background: #00b7ff;
        color: black;
        text-style: bold;
    }

    Button:focus {
        background: #00b7ff;
        color: black;
        text-style: bold;
    }

    #action-buttons {
        align-horizontal: center;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("\nVoronoi Configuration\n", id="title")

        with Center():
            yield VerticalScroll(
                # === BASIC INPUTS SECTION ===
                Static("BASIC INPUTS", id="basic-title"),

                Label("Input File:", classes="basic-label"),
                Input(value="experiment_2022_raw/0.csv", id="input_csv"),

                Label("Output Directory Name:", classes="basic-label"),
                Input(value="experiment_try0", id="output_dir"),

                Label("Sample Dimensions (x0,x1,y0,y1,z0,z1):", classes="basic-label"),
                Input(value="-477.0,528,-487,532,-1025,625", id="bbox"),

                Label("Sample Rotate Angles (rad):", classes="basic-label"),
                Input(value=f"0,0,{-(3.6/180)*math.pi:.6f}", id="rotate_angles"),

                Label("Rotation Unit (deg/rad):", classes="basic-label"),
                Input(value="rad", id="rotate_unit"),

                Label("Orientation Angle Identifier (comma separated):", classes="basic-label"),
                Input(value="Eul0,Eul1,Eul2", id="angle_identifier"),

                Label("Elastic Strain Identifiers (row-major ordered, comma separated):", classes="basic-label"),
                Input(value="eKen11,eKen12,eKen13,eKen21,eKen22,eKen23,eKen31,eKen32,eKen33", id="elastic_ids"),

                Label("Generate Mesh (True/False):", classes="basic-label"),
                Input(value="False", id="generate_mesh"),

                # === ADVANCED INPUTS SECTION ===
                Static("ADVANCED INPUTS", id="advanced-title"),

                Label("Orientation Descriptor:", classes="advanced-label"),
                Input(value="euler-bunge", id="orientation_descriptor"),

                Label("Orientation Active Convention (True/False):", classes="advanced-label"),
                Input(value="True", id="orientation_active_convention"),

                Label("Mesh Quality Min:", classes="advanced-label"),
                Input(value="0.7", id="mesh_quality_min"),

                Label("Relative Element Size:", classes="advanced-label"),
                Input(value="5.0", id="relative_el_size"),

                Label("Option ('voronoi','centroid','centroidsize'):", classes="advanced-label"),
                Input(value="centroid", id="option"),

                Label("CVT Iterations:", classes="advanced-label"),
                Input(value="100", id="CVT_iter"),

                Label("Morphological Algorithm ('subplex','lloyd','praxis'):", classes="advanced-label"),
                Input(value="subplex", id="morphoalgo"),
            )

        # Fixed action buttons
        with Center():
            with Horizontal(id="action-buttons"):
                yield Button("Run Voronoi", id="run")
                yield Button("Back to Menu", id="back")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id
        if btn == "back":
            self.app.pop_screen()
        elif btn == "run":
            self.launch_voronoi_process()

    def launch_voronoi_process(self) -> None:
        """Start Voronoi mesh builder in background via nohup."""
        from tui.log_viewer import LogViewerScreen

        try:
            # Prevent duplicate runs
            if self.current_proc and self.current_proc.poll() is None:
                self.app.notify("A Voronoi run is already active.")
                return
        
            def val(id): return self.query_one(f"#{id}", Input).value

            outputdir = val("output_dir")
            os.makedirs(outputdir, exist_ok=True)
            log_path = os.path.join(outputdir, "voronoi_run.log")

            params = {
                "input_csv": val("input_csv"),
                "output_dir": outputdir,
                "bounding_box": [float(x) for x in val("bbox").split(",")],
                "dim": 3,
                "weighted": False,
                "auto_fix_bbox": True,
                "bbox_fix_mode": "remove_points",
                "bbox_tolerance": 0.0,
                "auto_rotate": False,
                "rotate_angles": [float(a) for a in val("rotate_angles").split(",")],
                "rotate_convention": "xyz",
                "unit": val("rotate_unit"),
                "angle_identifier": [s.strip() for s in val("angle_identifier").split(",")],
                "orientation_descriptor": val("orientation_descriptor"),
                "orientation_active_convention": val("orientation_active_convention").lower() == "true",
                "elastic_strain_identifier": [s.strip() for s in val("elastic_ids").split(",")],
                "strain_unit": "microstrain",
                "generate_mesh": val("generate_mesh").lower() == "true",
                "mesh_quality_min": float(val("mesh_quality_min")),
                "relative_el_size": float(val("relative_el_size")),
                "option": val("option"),
                "CVT_iter": int(val("CVT_iter")),
                "morphoalgo": val("morphoalgo"),
            }

            params_json = json.dumps(params)

            # Launch the external process (detached)
            from pathlib import Path
            run_script = Path(__file__).parent / "run_file" / "run_voronoi_job.py"

            with open(log_path, "w") as log_file:
                self.current_proc = subprocess.Popen(
                    ["nohup", "python", "-u", str(run_script)],
                    stdin=subprocess.PIPE,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    preexec_fn=os.setpgrp,  
                )
                self.current_proc.stdin.write(params_json)
                self.current_proc.stdin.close()

            self.app.push_screen(LogViewerScreen(log_path, self.current_proc))
            self.app.notify(f"Voronoi job started, the output screen is also saved in {log_path}.")

        except Exception as e:
            self.app.notify(f"Error: {e}")

    def terminate_voronoi_process(self) -> None:
        """Safely stop the background Voronoi process."""
        try:
            if self.current_proc and self.current_proc.poll() is None:
                os.killpg(os.getpgid(self.current_proc.pid), signal.SIGTERM)
                self.app.notify("Voronoi run terminated.")
                self.current_proc = None
            else:
                self.app.notify("No active Voronoi run.")
        except Exception as e:
            self.app.notify(f"Error stopping run: {e}")
