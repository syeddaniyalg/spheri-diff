import ast
import io
import json
import os
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest
import yaml

from spheridiff.common.config import CONFIG_ROOT, PROJECT_ROOT, load_data_config
from spheridiff.common.seed import DEFAULT_SEED
from spheridiff.data.channels import channel_names
from spheridiff.evaluation.protocol import load_protocol
from spheridiff.evaluation.settings import load_full_settings
from spheridiff.experiments.spec import ExperimentSpec, load_experiment
from spheridiff.smoke.stages import STAGES
from spheridiff.training.settings import load_train_settings

CODE_FOLDERS = ("src", "scripts", "tests")
TEXT_FOLDERS = ("src", "scripts", "tests", "configs", "docs", "notebooks")
FORBIDDEN_NAME = "SI" + "ND"
AUTO_PROGRESS = "tqdm" + ".auto"
EM_DASH = chr(0x2014)
NOTEBOOK = PROJECT_ROOT / "notebooks" / "kaggle_smoke.ipynb"
PILOT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "kaggle_pilot.ipynb"
TIMEMATCHED_NOTEBOOK = PROJECT_ROOT / "notebooks" / "kaggle_pilot_timematched.ipynb"
NOTEBOOKS = [NOTEBOOK, PILOT_NOTEBOOK, TIMEMATCHED_NOTEBOOK]


def python_files():
    for folder in CODE_FOLDERS:
        yield from (PROJECT_ROOT / folder).rglob("*.py")


def text_files():
    for folder in TEXT_FOLDERS:
        for path in (PROJECT_ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".md", ".ipynb", ".toml"}:
                yield path
    yield PROJECT_ROOT / "README.md"


def comment_tokens(source):
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [token for token in tokens if token.type == tokenize.COMMENT]


@pytest.mark.parametrize("path", list(python_files()), ids=lambda path: str(path.relative_to(PROJECT_ROOT)))
def test_python_files_have_no_comments_or_docstrings(path):
    source = path.read_text(encoding="utf-8")
    assert comment_tokens(source) == []
    tree = ast.parse(source)
    nodes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert [node for node in nodes if ast.get_docstring(node)] == []
    assert AUTO_PROGRESS not in source


def test_text_files_avoid_the_deprecated_name_and_em_dashes():
    for path in text_files():
        text = path.read_text(encoding="utf-8")
        assert FORBIDDEN_NAME not in text, path
        assert EM_DASH not in text, path


def test_every_source_folder_is_a_package():
    for folder in (PROJECT_ROOT / "src" / "spheridiff").rglob("*"):
        if folder.is_dir() and folder.name != "__pycache__":
            assert (folder / "__init__.py").exists(), folder


def notebook_source(path):
    return "\n".join("".join(cell["source"]) for cell in json.loads(path.read_text())["cells"])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_is_valid_and_follows_the_style_rules(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code
    for cell in code:
        source = "".join(cell["source"])
        compile(source, "cell", "exec")
        assert comment_tokens(source) == []
        assert AUTO_PROGRESS not in source


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebooks_package_the_results_even_after_a_failure(path):
    source = notebook_source(path)
    run_cell = next(
        "".join(cell["source"])
        for cell in json.loads(path.read_text())["cells"]
        if cell["cell_type"] == "code" and "ParallelRunner(" in "".join(cell["source"])
    )
    assert run_cell.index("try:") < run_cell.index("RUNNER.run()") < run_cell.index("finally:")
    assert run_cell.index("finally:") < run_cell.index("finalise(")
    assert "CUDA_VISIBLE_DEVICES" not in source
    assert "assign_queues" in source and "refresh_archive" in source


def test_smoke_notebook_refers_to_existing_files_and_experiments():
    source = notebook_source(NOTEBOOK)
    assert (PROJECT_ROOT / "scripts" / "smoke.py").exists()
    for name in ("smoke_wind_s", "smoke_spheridiff_b4"):
        assert name in source
        assert (CONFIG_ROOT / "experiments" / f"{name}.yaml").exists()
    for stage in STAGES:
        assert f'"{stage}"' in source


def test_pilot_notebooks_run_the_final_evaluation_after_the_training():
    for path in (PILOT_NOTEBOOK, TIMEMATCHED_NOTEBOOK):
        source = notebook_source(path)
        assert "final_job(" in source and "SESSION_HOURS" not in source
        assert source.index("RUNNER.run()") < source.index("EVALUATOR.run()") < source.index("finally:")
        assert "final_names=" in source and "DEADLINE" not in source
    assert "final_job" not in notebook_source(NOTEBOOK)


def test_the_final_protocol_fields_exist_in_the_channel_set():
    names = channel_names(load_data_config("era5_5p6deg").channels)
    protocol = load_protocol("final_v1")
    assert set(protocol.fields) <= set(names)
    assert protocol.split == "test" and protocol.split not in ("train", "val")


def test_time_matched_notebook_is_separate_and_compares_the_final_errors():
    source = notebook_source(TIMEMATCHED_NOTEBOOK)
    assert "[[train_job(RUN_NAME)]]" in source
    assert 'RUN_NAME = "pilot_wind_l_long"' in source
    assert "pilot_timematched_results.zip" in source
    assert "reference_root=REFERENCE_ROOT" in source and "at_final=True" in source
    assert (CONFIG_ROOT / "experiments" / "pilot_wind_l_long.yaml").exists()
    assert "smoke" not in source


def test_time_matched_config_changes_only_the_length():
    pilot, long = load_train_settings("pilot"), load_train_settings("pilot_long")
    assert long.steps == 3 * pilot.steps
    assert long.evaluate_every == 3 * pilot.evaluate_every
    assert long.checkpoint_every == 3 * pilot.checkpoint_every
    for name in ("batch_size", "learning_rate", "min_learning_rate", "warmup_fraction", "weight_decay"):
        assert getattr(long, name) == getattr(pilot, name)
    for name in ("gradient_clip", "ema_decay", "precision", "milestones", "evaluation", "source_levels"):
        assert getattr(long, name) == getattr(pilot, name)
    spec = load_experiment("pilot_wind_l_long")
    assert (spec.model, spec.train, spec.evaluation, spec.seed) == ("wind_l", "pilot_long", "pilot", DEFAULT_SEED)


def test_pilot_notebook_runs_wind_and_spheridiff_on_separate_queues():
    source = notebook_source(PILOT_NOTEBOOK)
    for name in ("pilot_wind_s", "pilot_wind_m", "pilot_wind_l", "pilot_spheridiff_b4"):
        assert name in source
        assert (CONFIG_ROOT / "experiments" / f"{name}.yaml").exists()
    assert "[[train_job(name) for name in WIND_EXPERIMENTS], [train_job(name) for name in SPHERIDIFF_EXPERIMENTS]]" in source


@pytest.mark.parametrize("path", sorted(CONFIG_ROOT.rglob("*.yaml")), ids=lambda path: path.stem)
def test_configs_parse(path):
    assert isinstance(yaml.safe_load(path.read_text()), dict)


def test_experiments_resolve_to_existing_configs():
    for path in (CONFIG_ROOT / "experiments").glob("*.yaml"):
        spec = load_experiment(path.stem)
        for folder, name in (("data", spec.data), ("model", spec.model), ("train", spec.train)):
            assert (CONFIG_ROOT / folder / f"{name}.yaml").exists(), path
        assert (CONFIG_ROOT / "evaluation" / f"{spec.evaluation}.yaml").exists()
        assert (CONFIG_ROOT / "train" / f"{spec.pilot_train}.yaml").exists()


def test_evaluation_fields_exist_in_the_channel_set():
    names = channel_names(load_data_config("era5_5p6deg").channels)
    for stem in ("pilot", "smoke", "overfit"):
        assert set(load_train_settings(stem).evaluation.fields) <= set(names)
    for stem in ("pilot", "smoke"):
        assert set(load_full_settings(stem).fields) <= set(names)


def test_every_experiment_uses_the_standard_seed():
    assert DEFAULT_SEED == 42
    for path in (CONFIG_ROOT / "experiments").glob("*.yaml"):
        assert yaml.safe_load(path.read_text()).get("seed", DEFAULT_SEED) == DEFAULT_SEED, path
    for path in (CONFIG_ROOT / "train").glob("*.yaml"):
        evaluation = yaml.safe_load(path.read_text())["evaluation"]
        assert evaluation.get("seed", DEFAULT_SEED) == DEFAULT_SEED, path
    for path in (CONFIG_ROOT / "evaluation").glob("*.yaml"):
        assert yaml.safe_load(path.read_text()).get("seed", DEFAULT_SEED) == DEFAULT_SEED, path


def test_defaults_follow_the_standard_seed():
    assert load_experiment("pilot_wind_s").seed == DEFAULT_SEED
    assert load_train_settings("pilot").evaluation.seed == DEFAULT_SEED
    assert load_full_settings("pilot").seed == DEFAULT_SEED
    spec = ExperimentSpec(name="n", data="d", model="m", train="t", evaluation="e")
    assert spec.seed == DEFAULT_SEED


def test_loss_levels_cover_the_data_levels():
    data = load_data_config("era5_5p6deg")
    levels = set(data.channels.levels)
    for stem in ("pilot", "smoke", "overfit"):
        assert levels <= set(load_train_settings(stem).source_levels)


def test_command_line_scripts_start():
    environment = {
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    for script in (
        "train",
        "evaluate",
        "smoke",
        "plot_runs",
        "compare_runs",
        "package_runs",
        "final_evaluation",
        "final_summary",
        "prepare_data",
    ):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / f"{script}.py"), "--help"],
            capture_output=True,
            text=True,
            env={**os.environ, **environment},
        )
        assert result.returncode == 0, (script, result.stderr)
