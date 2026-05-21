from __future__ import annotations

import os, signal, time
from pathlib import Path
from textual.app import ComposeResult
from textual.widgets import Header, Footer, Static, Button, Log
from textual.containers import Center, Horizontal
from textual.screen import Screen


class LogViewerScreen(Screen):
    """Scrollable log viewer with live updates, heartbeat, and termination control."""

    CSS = """
    Screen {
        align: center middle;
    }

    #title {
        text-align: center;
        background: blue 50%;
        border: wide red;
        color: white;
        margin: 1 10;
    }

    Log {
        width: 120;
        height: 40;
        border: round yellow;
        background: black;
        color: ansi_bright_white;
        margin: 2;
    }

    Button {
        width: 25;
        margin: 1 2;
    }
    """

    def __init__(self, log_path: str, proc=None):
        super().__init__()
        self.log_path = Path(log_path)
        self.proc = proc
        self.status_line = ""
        self._heartbeat_counter = 0  # seconds since last heartbeat

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(f"\nVoronoi Run Log\n{self.log_path}\n", id="title")

        self.textlog = Log(auto_scroll=True)
        yield Center(self.textlog)

        with Center():
            with Horizontal():
                yield Button("Refresh", id="refresh")
                yield Button("Terminate Run", id="terminate")
                yield Button("Back to Config", id="back")

        yield Footer()

    def on_mount(self) -> None:
        """Set up periodic refresh and heartbeat timers."""
        self.load_log()
        # Refresh view twice per second
        self.set_interval(0.5, self.auto_refresh)
        # Heartbeat every second
        self.set_interval(1.0, self.heartbeat_tick)

    # ----------------------------
    # Periodic behaviors
    # ----------------------------
    def auto_refresh(self) -> None:
        """Periodic refresh that runs while screen is active."""
        self.load_log(live=True)

    def heartbeat_tick(self) -> None:
        """Append 'still running' message every certain seconds if process is alive."""
        if not self.proc or self.proc.poll() is not None:
            return  # no active process

        self._heartbeat_counter += 1
        if self._heartbeat_counter >= 120:
            self._heartbeat_counter = 0
            try:
                with open(self.log_path, "a") as f:
                    f.write("Program is still running...\n")
                    f.flush()
            except Exception:
                pass  # ignore transient file issues

    # ----------------------------
    # Core logic
    # ----------------------------
    def load_log(self, live: bool = False) -> None:
        """Load or refresh log content without losing status footer."""
        try:
            if self.log_path.exists():
                content = self.log_path.read_text()
                self.textlog.clear()
                self.textlog.write(content)
            else:
                self.textlog.clear()
                self.textlog.write(f"(No log file found at {self.log_path})")

            if self.status_line:
                self.textlog.write(f"\n{self.status_line}")
        except Exception as e:
            self.textlog.write(f"\nError reading log: {e}")

    def terminate_run(self) -> None:
        """Safely stop the background process with feedback delay."""
        try:
            if self.proc and self.proc.poll() is None:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.status_line = "[Terminating process ...]"
                self.load_log()

                # Wait up to 10 seconds for graceful shutdown
                for _ in range(10):
                    time.sleep(1.0)
                    if self.proc.poll() is not None:
                        break

                self.status_line = "[Terminated process successfully.]"
                self.proc = None
                self.load_log()
            else:
                self.status_line = "[No active process to terminate.]"
                self.load_log()
        except Exception as e:
            self.status_line = f"[Error stopping process: {e}]"
            self.load_log()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "refresh":
                self.load_log()
            case "terminate":
                self.terminate_run()
            case "back":
                self.app.pop_screen()
