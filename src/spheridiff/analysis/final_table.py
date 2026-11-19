import json
from pathlib import Path

RESULT_NAME = "final_evaluation.json"
FORECAST_KEYS = ("nrmse/mean", "crps_normalised/mean")


def load_result(root, name):
    path = Path(root) / name / RESULT_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def collect(groups):
    results = []
    for root, names in groups:
        for name in names:
            result = load_result(root, name)
            if result is not None:
                results.append((name, result))
    return results


def task_columns(results):
    first = results[0][1]["tasks"]
    return [(task, method) for task, methods in first.items() for method in methods if method != "baseline"]


def task_value(result, task, method):
    entry = result["tasks"][task][method]
    return entry["nrmse_unobserved"]


def result_row(name, result, columns):
    run = result["run"]
    row = {
        "run": name,
        "parameters": run["parameters"],
        "steps": run["trained_steps"],
        "complete": run["training_complete"],
    }
    row["forecast_nrmse"] = result["forecast"][FORECAST_KEYS[0]]
    row["forecast_ncrps"] = result["forecast"][FORECAST_KEYS[1]]
    for task, method in columns:
        row[f"{task}/{method}"] = task_value(result, task, method)
    return row


def baseline_row(results, columns):
    row = {"run": "task reference", "parameters": "", "steps": "", "complete": ""}
    row["forecast_nrmse"] = row["forecast_ncrps"] = ""
    first = results[0][1]
    for task, method in columns:
        row[f"{task}/{method}"] = task_value(first, task, "baseline")
    return row


def verify_protocol(results):
    digests = {result["protocol"]["digest"] for _, result in results}
    if len(digests) > 1:
        raise ValueError(f"the results come from different protocols: {sorted(digests)}")
    windows = {json.dumps(result["windows"], sort_keys=True) for _, result in results}
    if len(windows) > 1:
        raise ValueError("the results were computed on different evaluation windows")


def build_rows(results):
    verify_protocol(results)
    columns = task_columns(results)
    rows = [result_row(name, result, columns) for name, result in results]
    return rows + [baseline_row(results, columns)]


def format_cell(value):
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def format_table(rows):
    headers = list(rows[0])
    cells = [[format_cell(row[header]) for header in headers] for row in rows]
    widths = [max(len(header), *(len(line[index]) for line in cells)) for index, header in enumerate(headers)]
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths))]
    lines += ["  ".join(cell.ljust(width) for cell, width in zip(line, widths)) for line in cells]
    return "\n".join(lines)


def write_final_summary(output_root, groups):
    results = collect(groups)
    if not results:
        return None
    rows = build_rows(results)
    protocol = results[0][1]["protocol"]
    header = f"protocol {protocol['name']} ({protocol['digest'][:12]}), split {protocol['split']}"
    note = "forecast columns: normalised RMSE and CRPS over leads; task columns: normalised RMSE on unobserved entries"
    text = "\n".join([header, note, "", format_table(rows)])
    root = Path(output_root)
    (root / "final_evaluation_summary.txt").write_text(text, encoding="utf-8")
    (root / "final_evaluation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return text
