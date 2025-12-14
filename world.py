"""
Drive a random canvas agent that plays with letter canvases and periodically saves the
best matching output/reference pair.
"""

import argparse
import subprocess
from pathlib import Path
import time
from typing import Optional, List, Dict, Any

from PIL import Image

from agent import RandomCanvasAgent
import trackio as wandb


def save_best_pair(
    best: dict, output_path: Path, window_start: int, window_end: int
) -> Optional[Path]:
    """
    Persist the best comparison result for a window as a side-by-side PNG.
    """
    if not best:
        return None

    ref_img = best["reference_img"]
    out_img = best["output_img"]
    combined = Image.new(
        "RGB",
        (ref_img.width + out_img.width, max(ref_img.height, out_img.height)),
        color="white",
    )
    combined.paste(ref_img, (0, 0))
    combined.paste(out_img, (ref_img.width, 0))

    meta = best.get("metadata") or {}
    letter = meta.get("letter", "?")
    idx = meta.get("index")
    idx_str = f"idx_{idx}_" if idx is not None else ""
    filename = (
        f"best_cmp_{window_start:06d}_{window_end:06d}_{idx_str}"
        f"{letter}_mse_{best['loss']:.2f}.png"
    )
    combined_path = output_path / filename
    combined.save(combined_path)
    return combined_path


def run(
    steps: Optional[int] = None,
    window_size: int = 1,
    delay: float = 0.0,
    output_dir: str = "world_output",
    seed: Optional[int] = None,
    open_viewer: bool = False,
    viewer_best: int = 5,
    viewer_refresh_ms: int = 500,
) -> None:
    """
    Drive a random agent and save the lowest-MSE comparison every window.
    """
    print(
        "Running random canvas agent (reference/state/output). "
        "Saves lowest MSE comparison every window_size comparisons. Ctrl+C to stop."
    )

    output_path = Path(output_dir)
    if output_path.exists():
        for child in output_path.glob("*"):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                for sub in child.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                child.rmdir()
    output_path.mkdir(parents=True, exist_ok=True)

    agent = RandomCanvasAgent(seed=seed)

    window_best = None
    window_compare_count = 0
    window_start_compare = 1
    total_compares = 0
    actions_run = 0
    best_overall: Dict[str, Any] = {"loss": float("inf"), "path": None}
    best_list: List[Dict[str, Any]] = []
    best_dir = output_path / "best_live"
    best_dir.mkdir(exist_ok=True, parents=True)

    viewer_proc = None
    if open_viewer:
        cmd = [
            "python",
            "viewer.py",
            "--watch-dir",
            str(output_path),
            "--best-dir",
            str(best_dir),
            "--refresh-ms",
            str(viewer_refresh_ms),
        ]
        viewer_proc = subprocess.Popen(cmd)

    def save_png_atomic(img, path: Path):
        # Keep the real extension so Pillow knows the format (e.g., .png)
        tmp = path.with_name(path.stem + ".tmp" + path.suffix)
        img.save(tmp)
        tmp.replace(path)

    def save_live_images():
        if agent.reference_canvas is None or agent.output_canvas is None:
            return
        ref_img = agent.reference_canvas.to_pil()
        out_img = agent.output_canvas.to_pil()
        state_img = agent.state_canvas.to_pil() if agent.state_canvas else None
        save_png_atomic(ref_img, output_path / "live_reference.png")
        save_png_atomic(out_img, output_path / "live_output.png")
        if state_img is not None:
            save_png_atomic(state_img, output_path / "live_state.png")

    def update_best_list(loss: float, meta: dict, step_idx: int, compare_idx: int):
        if agent.reference_canvas is None or agent.output_canvas is None:
            return
        ref_img = agent.reference_canvas.to_pil()
        out_img = agent.output_canvas.to_pil()
        combined = Image.new(
            "RGB",
            (ref_img.width + out_img.width, max(ref_img.height, out_img.height)),
            color="white",
        )
        combined.paste(ref_img, (0, 0))
        combined.paste(out_img, (ref_img.width, 0))
        entry = {
            "loss": loss,
            "meta": meta,
            "step": step_idx,
            "compare_idx": compare_idx,
            "image": combined,
        }
        inserted = False
        for i, e in enumerate(best_list):
            if loss < e["loss"]:
                best_list.insert(i, entry)
                inserted = True
                break
        if not inserted:
            best_list.append(entry)
        # trim
        if len(best_list) > viewer_best:
            removed = best_list[viewer_best:]
            best_list[:] = best_list[:viewer_best]
            for r in removed:
                r_path = r.get("path")
                if r_path and Path(r_path).exists():
                    Path(r_path).unlink(missing_ok=True)
        # rewrite files with ranking
        for idx, e in enumerate(best_list, start=1):
            meta_letter = (e.get("meta") or {}).get("letter", "?")
            filename = (
                f"best_{idx:02d}_step_{e['step']:06d}_cmp_{e['compare_idx']:06d}_"
                f"{meta_letter}_mse_{e['loss']:.2f}.png"
            )
            target_path = best_dir / filename
            save_png_atomic(e["image"], target_path)
            e["path"] = target_path
        if best_list:
            best_overall["loss"] = best_list[0]["loss"]
            best_overall["path"] = best_list[0]["path"]

    try:
        while steps is None or actions_run < steps:
            result = agent.step()
            actions_run += 1
            action = result["action"]
            step_idx = result["step_index"]

            if action == "compare":
                loss = result["loss"]
                meta = result.get("metadata") or {}
                letter = meta.get("letter", "?")
                ref_idx = meta.get("index")
                total_compares += 1
                window_compare_count += 1
                idx_text = f" (idx {ref_idx})" if ref_idx is not None else ""
                print(
                    f"[step {step_idx}] compare vs letter '{letter}'{idx_text} -> MSE {loss:.2f}"
                )

                if window_best is None or loss < window_best["loss"]:
                    if agent.reference_canvas is None or agent.output_canvas is None:
                        continue
                    ref_img = agent.reference_canvas.to_pil()
                    out_img = agent.output_canvas.to_pil()
                    window_best = {
                        "loss": loss,
                        "reference_img": ref_img,
                        "output_img": out_img,
                        "metadata": meta,
                        "step_index": step_idx,
                        "compare_index": total_compares,
                    }
                update_best_list(loss, meta, step_idx, total_compares)
                if loss < best_overall["loss"]:
                    if agent.reference_canvas is not None and agent.output_canvas is not None:
                        ref_img = agent.reference_canvas.to_pil()
                        out_img = agent.output_canvas.to_pil()
                        combined = Image.new(
                            "RGB",
                            (ref_img.width + out_img.width, max(ref_img.height, out_img.height)),
                            color="white",
                        )
                        combined.paste(ref_img, (0, 0))
                        combined.paste(out_img, (ref_img.width, 0))
                        meta_letter = meta.get("letter", "?")
                        filename = (
                            f"best_overall_step_{step_idx:06d}_cmp_{total_compares:06d}_"
                            f"{meta_letter}_mse_{loss:.2f}.png"
                        )
                        best_path = output_path / filename
                        combined.save(best_path)
                        best_overall = {"loss": loss, "path": best_path}
                        print(f"  [BEST] New best overall MSE {loss:.2f} saved to {best_path}")

                if window_compare_count == window_size:
                    saved_path = save_best_pair(
                        window_best, output_path, window_start_compare, total_compares
                    )
                    if saved_path:
                        print(
                            f"  Saved best comparisons {window_start_compare}-{total_compares} to {saved_path}"
                        )
                    window_best = None
                    window_compare_count = 0
                    window_start_compare = total_compares + 1
            else:
                if action.startswith("add_rect"):
                    rect = result.get("rect")
                    color = result.get("color")
                    print(f"[step {step_idx}] {action} rect={rect} color={color}")
                elif action in {"request_reference", "reset_canvases"}:
                    meta = result.get("metadata") or agent.reference_metadata or {}
                    letter = meta.get("letter")
                    ref_idx = meta.get("index")
                    extra = f" letter '{letter}'" if letter else ""
                    if ref_idx is not None:
                        extra += f" (idx {ref_idx})"
                    print(f"[step {step_idx}] {action}{extra}")
                else:
                    print(f"[step {step_idx}] {action}")

            if open_viewer:
                save_live_images()

            wandb.log(
                {
                    "step_index": step_idx,
                    "action": action,
                    "mse_loss": result.get("loss", 0.0),
                    "reward": result.get("reward", 0.0),
                    "reward_base": result.get("reward_base", 0.0),
                    "reward_compare_action": result.get("reward_compare_action", 0.0),
                    "reward_compare_value": result.get("reward_compare_value", 0.0),
                    "reward_change": result.get("reward_change", 0.0),
                    "reward_occupancy": result.get("reward_occupancy", 0.0),
                    "action_entropy": result.get("entropy", 0.0),
                    "best_overall_mse": best_overall["loss"],
                }
            )

            if delay > 0:
                time.sleep(delay)
    finally:
        if window_best is not None and window_compare_count > 0:
            window_end = window_start_compare + window_compare_count - 1
            saved_path = save_best_pair(
                window_best, output_path, window_start_compare, window_end
            )
            if saved_path:
                print(
                    f"Saved best of final {window_compare_count} comparisons "
                    f"(window {window_start_compare}-{window_end}) to {saved_path}"
                )
        if viewer_proc is not None:
            viewer_proc.terminate()
            viewer_proc.wait(timeout=2)


if __name__ == "__main__":
    wandb.init(project="random_canvas_agent")

    parser = argparse.ArgumentParser(
        description="Generate random letter canvases and compare them to rectangle canvases."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of agent steps (0 runs indefinitely).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to pause between iterations.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Save the lowest MSE pair every N samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="world_output",
        help="Directory to write side-by-side PNGs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for the random agent.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the live viewer that watches output_dir for updates.",
    )
    parser.add_argument(
        "--viewer-best",
        type=int,
        default=5,
        help="How many best comparisons to keep in the live viewer list.",
    )
    parser.add_argument(
        "--viewer-refresh",
        type=int,
        default=500,
        help="Viewer refresh interval in milliseconds (polling).",
    )

    args = parser.parse_args()
    steps = None if args.limit == 0 else args.limit

    try:
        run(
            steps=steps,
            delay=args.delay,
            window_size=args.window_size,
            output_dir=args.output_dir,
            seed=args.seed,
            open_viewer=args.viewer,
            viewer_best=args.viewer_best,
            viewer_refresh_ms=args.viewer_refresh,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
