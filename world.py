"""
Drive a random canvas agent that plays with letter canvases and periodically saves the
best matching output/reference pair.
"""
import argparse
from pathlib import Path
import time
from typing import Optional

from PIL import Image

from agent import RandomCanvasAgent


def save_best_pair(best: dict, output_path: Path, window_start: int, window_end: int) -> Optional[Path]:
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
) -> None:
    """
    Drive a random agent and save the lowest-MSE comparison every window.
    """
    print(
        "Running random canvas agent (reference/state/output). "
        "Saves lowest MSE comparison every window_size comparisons. Ctrl+C to stop."
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    agent = RandomCanvasAgent(seed=seed)

    window_best = None
    window_compare_count = 0
    window_start_compare = 1
    total_compares = 0
    actions_run = 0

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
                print(f"[step {step_idx}] compare vs letter '{letter}'{idx_text} -> MSE {loss:.2f}")

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

                if window_compare_count == window_size:
                    saved_path = save_best_pair(window_best, output_path, window_start_compare, total_compares)
                    if saved_path:
                        print(f"  Saved best comparisons {window_start_compare}-{total_compares} to {saved_path}")
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

            if delay > 0:
                time.sleep(delay)
    finally:
        if window_best is not None and window_compare_count > 0:
            window_end = window_start_compare + window_compare_count - 1
            saved_path = save_best_pair(window_best, output_path, window_start_compare, window_end)
            if saved_path:
                print(
                    f"Saved best of final {window_compare_count} comparisons "
                    f"(window {window_start_compare}-{window_end}) to {saved_path}"
                )


if __name__ == "__main__":
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

    args = parser.parse_args()
    steps = None if args.limit == 0 else args.limit

    try:
        run(
            steps=steps,
            delay=args.delay,
            window_size=args.window_size,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
