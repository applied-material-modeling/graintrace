from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Button
from textual.containers import Vertical, Center

# import the separate screen
from tui.voronoi import VoronoiScreen


class GrainTRACE(App):
    """Main GrainTRACE Textual interface"""

    DARK = True

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

    Button {
        content-align: center middle;
        text-align: center;
        margin: 1 30;
        width: 50;
    }

    Button:hover {
        background: #00b7ff;
        color: black;
        text-style: bold;
    }

    Button:focus {
        color: yellow;
        text-style: bold;
    }

    .highlight {
        color: black;
        text-style: bold;
    }

    .highlight:hover, .highlight:focus {
        color: black;
        text-style: bold;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(
            "\nGrainTRACE\n\nGrain Texture and Response Analysis for Crystal Exploration\n",
            id="title",
        )

        with Center():
            yield Vertical(
                Button("FULL ANALYSIS PIPELINE", id="main"),
                Button("Build Voronoi", id="voronoi"),
                Button("Build Graph from Tesselation", id="graph"),
                Button("Calibrate Material", id="calibrate"),
                Button("Run Crystal Plasticity Approximation Model", id="cp_model"),
                Button("Run CPFE simulation", id="cpfe"),
                Button("Exit", id="exit"),
            )

        yield Footer()

    def on_ready(self) -> None:
        # Register the Voronoi configuration screen
        self.install_screen(VoronoiScreen(), name="voronoi")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id

        if btn == "exit":
            self.exit()
        elif btn == "voronoi":
            self.push_screen("voronoi")
        elif btn == "main":
            self.notify("Running full analysis pipeline...")
        elif btn == "graph":
            self.notify("Building graph from tessellation...")
        elif btn == "calibrate":
            self.notify("Calibrating material parameters...")
        elif btn == "cp_model":
            self.notify("Running Crystal Plasticity Approximation Model...")
        elif btn == "cpfe":
            self.notify("Running CPFE simulation...")

    def on_show(self) -> None:
        # Remove any initial focus highlight
        self.call_after_refresh(lambda: self.screen.set_focus(None))


if __name__ == "__main__":
    GrainTRACE().run()
