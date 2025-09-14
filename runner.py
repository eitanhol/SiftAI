import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

def run_chatbot(
    script_path: Path,
    csv_path: Path,
    question: str,
    *,
    out: str = "data_plan.json",
    model: str = "gemini-2.5-flash",
    no_sandbox: bool = False,
    run_sandbox: Optional[Path] = None,
    output: Optional[str] = None,              # "table" | "viz" | "both"
    params: Optional[Dict[str, Any]] = None,   # merged with --output on the Node side
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_sec: int = 900,
) -> Dict[str, Any]:
    """
    Run the Node CLI:
      node chatbot.js <csv_path_any> "<question>"
        [--no-sandbox]
        [--run-sandbox <csv_path>]
        [--out data_plan.json]
        [--model gemini-2.5-flash]
        [--output table|viz|both]
        [--params '{"output":"both"}']

    Returns a dict with:
      - returncode, stdout, stderr
      - plan (parsed JSON from data_plan.json if present)
      - result (parsed JSON from data_result.json if present)
      - result_png (Path if data_result.png exists)
      - result_html (Path if data_result.html exists)
      - cmd (the exact command run, for debugging)
    """
    script_path = Path(script_path)
    workdir = Path(cwd) if cwd else script_path.parent

    # Build the CLI
    cmd = ["node", str(script_path), str(csv_path), question]

    if no_sandbox and run_sandbox:
        raise ValueError("Provide either no_sandbox=True OR run_sandbox=..., not both.")

    if no_sandbox:
        cmd += ["--no-sandbox"]
    elif run_sandbox:
        cmd += ["--run-sandbox", str(run_sandbox)]

    if out:
        cmd += ["--out", out]
    if model:
        cmd += ["--model", model]
    if output:
        if output not in {"table", "viz", "both"}:
            raise ValueError("output must be one of {'table','viz','both'}")
        cmd += ["--output", output]
    if params is not None:
        # Pass as compact JSON; quoting handled by subprocess without shell=True
        cmd += ["--params", json.dumps(params, separators=(",", ":"))]

    # Execute
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        env=env,                 # inherit parent env (incl. GOOGLE_API_KEY) if None
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )

    # Fixed result filenames produced by chatbot.js
    result_json = workdir / "data_result.json"
    result_png  = workdir / "data_result.png"
    result_html = workdir / "data_result.html"
    plan_json   = workdir / (out or "data_plan.json")

    def _read_json(p: Path) -> Optional[dict]:
        try:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
        return None

    ret = {
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "plan": _read_json(plan_json),
        "result": _read_json(result_json),
        "result_png": result_png if result_png.exists() else None,
        "result_html": result_html if result_html.exists() else None,
    }

    return ret
