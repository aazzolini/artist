import argparse
import json
import os
import random
import socketserver
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import List, Tuple

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Canvas Policy Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            color-scheme: dark;
            --bg: #030711;
            --panel: rgba(10,16,33,0.9);
            --accent: #58ffcb;
            --accent-2: #8ab4ff;
            --text: #f5f7ff;
            --muted: #7e8aac;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Space Grotesk", system-ui, sans-serif;
            background: radial-gradient(circle at 20% 20%, #111832, var(--bg) 60%);
            color: var(--text);
            min-height: 100vh;
        }
        header {
            padding: 2rem 3vw 1rem;
        }
        h1 {
            margin: 0;
            font-size: 2.5rem;
            letter-spacing: 0.02em;
        }
        p { color: var(--muted); margin-top: 0.5rem; }
        main {
            padding: 1rem 3vw 3rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            grid-gap: 1.5rem;
        }
        .panel {
            background: var(--panel);
            border-radius: 18px;
            padding: 1.25rem;
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(255,255,255,0.05);
        }
        .panel h2 {
            margin: 0 0 0.5rem;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--muted);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
        }
        .stat {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }
        .stat-label { text-transform: uppercase; font-size: 0.75rem; color: var(--muted); }
        .stat-value { font-size: 1.8rem; margin: 0.4rem 0 0; font-weight: 600; }
        canvas { width: 100%; }
        footer { text-align: center; color: var(--muted); padding-bottom: 2rem; }
        .status-dot {
            width: 8px; height: 8px; border-radius: 999px; display: inline-block; margin-right: 0.35rem;
        }
        .status-online { background: var(--accent); }
        .status-offline { background: #ff8c8c; }
    </style>
</head>
<body>
    <header>
        <h1>Canvas Policy Monitor</h1>
        <p>Mini-dashboard inspired by wandb. Feed it via POST <code>/api/metrics</code> or append to the metrics file.</p>
    </header>
    <main>
        <section class="panel" style="grid-column: span 2; min-width: 300px;">
            <h2>Live Stats</h2>
            <div class="stats-grid">
                <div class="stat">
                    <div class="stat-label">Last Step</div>
                    <div class="stat-value" id="stat-step">--</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Current Loss</div>
                    <div class="stat-value" id="stat-loss">--</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Action Entropy</div>
                    <div class="stat-value" id="stat-entropy">--</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Update Status</div>
                    <div class="stat-value" style="font-size:1rem;">
                        <span class="status-dot status-offline" id="status-dot"></span>
                        <span id="status-text">Waiting</span>
                    </div>
                </div>
            </div>
        </section>
        <section class="panel">
            <h2>Loss (MSE)</h2>
            <canvas id="lossChart" height="220"></canvas>
        </section>
        <section class="panel">
            <h2>Action Entropy</h2>
            <canvas id="entropyChart" height="220"></canvas>
        </section>
    </main>
    <footer>POST JSON like {step, loss, entropy} to <code>/api/metrics</code>. Data refreshes every ~2s.</footer>
    <script>
    const lossCtx = document.getElementById('lossChart');
    const entropyCtx = document.getElementById('entropyChart');

    const lossChart = new Chart(lossCtx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Loss (MSE)', data: [], borderColor: '#4fa', fill: false }]},
        options: { responsive: true, scales: { x: { ticks: { color: '#bbb' }}, y: { ticks: { color: '#bbb'}}}}
    });

    const entropyChart = new Chart(entropyCtx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Action Entropy', data: [], borderColor: '#6bf', fill: false }]},
        options: { responsive: true, scales: { x: { ticks: { color: '#bbb' }}, y: { ticks: { color: '#bbb'}}}}
    });

    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const statStep = document.getElementById('stat-step');
    const statLoss = document.getElementById('stat-loss');
    const statEntropy = document.getElementById('stat-entropy');

    async function refresh() {
        try {
            const response = await fetch('/api/metrics');
            if (!response.ok) throw new Error('Failed to fetch metrics');
            const payload = await response.json();
            const steps = payload.map(d => d.step);
            const losses = payload.map(d => d.loss);
            const entropies = payload.map(d => d.entropy);
            if (payload.length) {
                const latest = payload[payload.length - 1];
                statStep.textContent = latest.step.toLocaleString();
                statLoss.textContent = latest.loss.toFixed(4);
                statEntropy.textContent = latest.entropy.toFixed(4);
                statusDot.classList.remove('status-offline');
                statusDot.classList.add('status-online');
                statusText.textContent = 'Streaming';
            } else {
                statusDot.classList.remove('status-online');
                statusDot.classList.add('status-offline');
                statusText.textContent = 'No data yet';
            }

            lossChart.data.labels = steps;
            lossChart.data.datasets[0].data = losses;
            lossChart.update('none');

            entropyChart.data.labels = steps;
            entropyChart.data.datasets[0].data = entropies;
            entropyChart.update('none');
        } catch (err) {
            statusDot.classList.remove('status-online');
            statusDot.classList.add('status-offline');
            statusText.textContent = 'Disconnected';
            console.error(err);
        }
        window.setTimeout(refresh, 2000);
    }

    refresh();
    </script>
</body>
</html>
"""


def read_metrics(path: Path, max_points: int) -> List[dict]:
    if not path.exists():
        return []
    data: List[dict] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = payload.get("step")
            loss = payload.get("loss")
            entropy = payload.get("entropy")
            if step is None or loss is None or entropy is None:
                continue
            data.append({"step": step, "loss": loss, "entropy": entropy})
    return data[-max_points:]


class MetricsHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, metrics_path: Path, max_points: int, **kwargs):
        self.metrics_path = metrics_path
        self.max_points = max_points
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            payload = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/api/metrics":
            data = read_metrics(self.metrics_path, self.max_points)
            payload = json.dumps(data).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self):  # noqa: N802
        if self.path != "/api/metrics":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return

        records = payload if isinstance(payload, list) else [payload]
        normalized: List[dict] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if not {"step", "loss", "entropy"} <= rec.keys():
                continue
            normalized.append(
                {"step": rec["step"], "loss": rec["loss"], "entropy": rec["entropy"]}
            )

        if not normalized:
            self.send_error(HTTPStatus.BAD_REQUEST, "No valid records provided")
            return

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a") as fh:
            for rec in normalized:
                fh.write(json.dumps(rec) + "\n")

        response = json.dumps({"stored": len(normalized)}).encode("utf-8")
        self.send_response(HTTPStatus.CREATED)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Silence default logging; stdout is used for port info only.
        return


def serve(metrics_path: Path, host: str, port: int, max_points: int) -> None:
    handler = lambda *args, **kwargs: MetricsHandler(  # noqa: E731
        *args,
        metrics_path=metrics_path,
        max_points=max_points,
        **kwargs,
    )

    with socketserver.ThreadingTCPServer((host, port), handler) as httpd:
        actual_host, actual_port = httpd.server_address
        print(f"Metrics server listening on http://{actual_host}:{actual_port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


def main():
    parser = argparse.ArgumentParser(description="Serve policy metrics dashboard.")
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="world_output/metrics.jsonl",
        help="Path to newline-delimited JSON metrics file.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", type=int, default=0, help="Port to bind (0 picks random).")
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Max number of metric points to return to clients.",
    )

    args = parser.parse_args()
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    serve(metrics_path=metrics_path, host=args.host, port=args.port, max_points=args.max_points)


if __name__ == "__main__":
    main()
