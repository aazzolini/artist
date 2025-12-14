"""
Simple Tkinter viewer that shows the reference, state, and output canvases side by side
and lets the user advance the RandomCanvasAgent one action at a time.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, List, Tuple
import argparse
from pathlib import Path

from PIL import Image, ImageTk

from agent import RandomCanvasAgent


DEFAULT_SIZE = 256
PNG_EXTS = {".png", ".jpg", ".jpeg"}


class CanvasViewer:
    """Tiny UI wrapper that wires a RandomCanvasAgent to a Tkinter window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Canvas Viewer")

        self.agent = RandomCanvasAgent()

        self.status_var = tk.StringVar(value="Click 'Next action' to start.")
        self.meta_var = tk.StringVar(value="")

        self.image_labels = {}
        self.photo_images = {}

        self._build_layout()
        self._init_reference()

    def _build_layout(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, textvariable=self.status_var).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(control_frame, text="Next action", command=self.on_next_action).pack(
            side=tk.LEFT
        )

        ttk.Label(control_frame, textvariable=self.meta_var, foreground="gray").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        canvas_frame = ttk.Frame(self.root, padding=10)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        for idx, name in enumerate(["Reference", "State", "Output"]):
            col = ttk.Frame(canvas_frame, padding=5)
            col.grid(row=0, column=idx, sticky="nsew")
            canvas_frame.columnconfigure(idx, weight=1)
            ttk.Label(col, text=name).pack()
            lbl = ttk.Label(col)
            lbl.pack()
            self.image_labels[name.lower()] = lbl

    def _init_reference(self):
        metadata = self.agent.request_new_reference()
        self.agent.reset_canvases()
        self._update_meta(metadata)
        self._refresh_images()
        self.status_var.set("Reference loaded. Ready for actions.")

    def _blank_image(self, width=DEFAULT_SIZE, height=DEFAULT_SIZE) -> Image.Image:
        return Image.new("RGB", (width, height), color="white")

    def _canvas_to_image(self, canvas):
        if canvas is None:
            return self._blank_image()
        return canvas.to_pil()

    def _refresh_images(self):
        ref_img = self._canvas_to_image(self.agent.reference_canvas)
        state_img = self._canvas_to_image(self.agent.state_canvas)
        out_img = self._canvas_to_image(self.agent.output_canvas)
        image_map = {
            "reference": ref_img,
            "state": state_img,
            "output": out_img,
        }
        for key, pil_img in image_map.items():
            photo = ImageTk.PhotoImage(pil_img)
            self.photo_images[key] = photo  # keep reference to prevent GC
            self.image_labels[key].configure(image=photo)

    def _update_meta(self, metadata: Optional[dict]):
        if not metadata:
            self.meta_var.set("")
            return
        letter = metadata.get("letter")
        idx = metadata.get("index")
        font = metadata.get("font_family")
        size = metadata.get("font_size")
        rotation = metadata.get("rotation")
        parts = []
        if letter is not None:
            parts.append(f"letter={letter}")
        if idx is not None:
            parts.append(f"idx={idx}")
        if font is not None:
            parts.append(f"font={font}")
        if size is not None:
            parts.append(f"size={size}")
        if rotation is not None:
            parts.append(f"rot={rotation:.1f}")
        self.meta_var.set(" | ".join(parts))

    def on_next_action(self):
        result = self.agent.step()
        action = result.get("action")
        meta = result.get("metadata") or self.agent.reference_metadata
        self._update_meta(meta)

        if action == "compare":
            loss = result.get("loss", 0.0)
            self.status_var.set(f"Action: {action} | MSE={loss:.2f}")
        elif action == "request_reference":
            self.status_var.set(f"Action: {action} -> loaded new letter")
            # ensure canvases exist for new reference
            self.agent.reset_canvases()
        else:
            self.status_var.set(f"Action: {action}")

        self._refresh_images()


class WatchedViewer:
    """
    Poll a directory for live canvas PNGs and an optional best-images directory.
    Expects files:
      watch_dir/live_reference.png
      watch_dir/live_state.png (optional)
      watch_dir/live_output.png
    And best_dir containing PNGs sorted lexicographically.
    """

    def __init__(
        self,
        root: tk.Tk,
        watch_dir: Path,
        best_dir: Optional[Path] = None,
        refresh_ms: int = 500,
    ):
        self.root = root
        self.watch_dir = Path(watch_dir)
        self.best_dir = Path(best_dir) if best_dir else None
        self.refresh_ms = max(100, refresh_ms)

        self.status_var = tk.StringVar(value="Watching directory...")
        self.best_status_var = tk.StringVar(value="Best: 0 items")
        self.meta_var = tk.StringVar(value="")

        self.image_labels = {}
        self.photo_images = {}

        self.best_images: List[Tuple[Path, Image.Image]] = []
        self.best_index = 0

        self._build_layout()

    def _build_layout(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(control_frame, textvariable=self.status_var).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(control_frame, textvariable=self.meta_var, foreground="gray").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        canvas_frame = ttk.Frame(self.root, padding=10)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        for idx, name in enumerate(["Reference", "State", "Output"]):
            col = ttk.Frame(canvas_frame, padding=5)
            col.grid(row=0, column=idx, sticky="nsew")
            canvas_frame.columnconfigure(idx, weight=1)
            ttk.Label(col, text=name).pack()
            lbl = ttk.Label(col)
            lbl.pack()
            self.image_labels[name.lower()] = lbl

        best_frame = ttk.Frame(self.root, padding=10)
        best_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(best_frame, text="Best list").pack(side=tk.LEFT)
        ttk.Button(best_frame, text="Prev", command=self.show_prev_best).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(best_frame, text="Next", command=self.show_next_best).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Label(best_frame, textvariable=self.best_status_var).pack(side=tk.LEFT)

        self.best_label = ttk.Label(self.root)
        self.best_label.pack()

    def start(self):
        self._tick()

    def _tick(self):
        self._refresh_live()
        self._refresh_best()
        self.root.after(self.refresh_ms, self._tick)

    def _load_image(self, path: Path) -> Optional[Image.Image]:
        if not path.exists():
            return None
        try:
            return Image.open(path).copy()
        except Exception:
            return None

    def _refresh_live(self):
        ref_path = self.watch_dir / "live_reference.png"
        state_path = self.watch_dir / "live_state.png"
        out_path = self.watch_dir / "live_output.png"
        ref_img = self._load_image(ref_path) or self._blank_image()
        state_img = self._load_image(state_path) or self._blank_image()
        out_img = self._load_image(out_path) or self._blank_image()
        image_map = {
            "reference": ref_img,
            "state": state_img,
            "output": out_img,
        }
        for key, pil_img in image_map.items():
            photo = ImageTk.PhotoImage(pil_img)
            self.photo_images[key] = photo
            self.image_labels[key].configure(image=photo)
        self.status_var.set(f"Watching {self.watch_dir}")

    def _refresh_best(self):
        if self.best_dir is None:
            self.best_status_var.set("Best: disabled")
            return
        candidates = sorted(
            [p for p in self.best_dir.iterdir() if p.suffix.lower() in PNG_EXTS]
        )
        if not candidates:
            self.best_images = []
            self.best_status_var.set("Best: 0 items")
            self.best_label.configure(image="")
            return
        if len(candidates) != len(self.best_images) or any(
            p != old[0] for p, old in zip(candidates, self.best_images)
        ):
            loaded: List[Tuple[Path, Image.Image]] = []
            for p in candidates:
                img = self._load_image(p)
                if img:
                    loaded.append((p, img))
            self.best_images = loaded
            # Always snap to best (first) when list changes
            self.best_index = 0
        self.best_status_var.set(
            f"Best: {len(self.best_images)} items (showing {self.best_index+1})"
        )
        self._show_best_current()

    def _show_best_current(self):
        if not self.best_images:
            self.best_label.configure(image="")
            return
        path, img = self.best_images[self.best_index]
        photo = ImageTk.PhotoImage(img)
        self.photo_images["best"] = photo
        self.best_label.configure(image=photo)
        mse_text = self._extract_mse(path.name)
        self.meta_var.set(f"{path.name} | MSE={mse_text}")

    def show_prev_best(self):
        if not self.best_images:
            return
        self.best_index = (self.best_index - 1) % len(self.best_images)
        self._show_best_current()

    def show_next_best(self):
        if not self.best_images:
            return
        self.best_index = (self.best_index + 1) % len(self.best_images)
        self._show_best_current()

    def _blank_image(self, width=DEFAULT_SIZE, height=DEFAULT_SIZE) -> Image.Image:
        return Image.new("RGB", (width, height), color="white")

    def _extract_mse(self, name: str) -> str:
        """
        Parse 'mse_XX.YY' from filename if present.
        """
        try:
            import re

            match = re.search(r"mse_([0-9]+\\.[0-9]+)", name)
            if match:
                return match.group(1)
        except Exception:
            pass
        return "?"

def main():
    parser = argparse.ArgumentParser(
        description="Canvas viewer. Either run with a live agent or watch a directory."
    )
    parser.add_argument(
        "--watch-dir",
        type=str,
        default=None,
        help="Directory to watch for live_reference.png/live_state.png/live_output.png",
    )
    parser.add_argument(
        "--best-dir",
        type=str,
        default=None,
        help="Directory to read rolling best images from (png).",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=500,
        help="Refresh interval in milliseconds for watch mode.",
    )
    args = parser.parse_args()

    root = tk.Tk()
    if args.watch_dir:
        watcher = WatchedViewer(
            root,
            watch_dir=Path(args.watch_dir),
            best_dir=Path(args.best_dir) if args.best_dir else None,
            refresh_ms=args.refresh_ms,
        )
        watcher.start()
    else:
        CanvasViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
