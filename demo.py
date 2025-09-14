from pathlib import Path
from runner import run_chatbot

resp = run_chatbot(
    script_path=Path("./chatbot.js"),
    csv_path=Path("./data.csv"),
    question="Total Rows",
    run_sandbox=Path("./data.csv"),
    output="both",
    params={"top_n": 20, "include_png": True},
)

print("Return code:", resp["returncode"])
print("Plan keys:", list((resp["plan"] or {}).keys()))
print("Result keys:", list((resp["result"] or {}).keys()))
print("PNG path:", resp["result_png"])
