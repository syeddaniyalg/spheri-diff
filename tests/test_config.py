from pathlib import Path

import pytest

from helpers import make_cache, make_config
from spheridiff.common.config import PROJECT_ROOT, load_data_config, resolve_config_path
from spheridiff.data.download import expected_steps, select_years
from spheridiff.data.loading import build_dataset
from spheridiff.data.plan import check_cache, read_bytes_per_year, steps_per_year
from spheridiff.data.statistics import build_statistics

CONFIG_NAME = "era5_5p6deg"


def test_config_is_found_by_name_and_by_path():
    by_name = resolve_config_path(CONFIG_NAME)
    by_path = resolve_config_path(str(by_name))
    assert by_name == by_path


def test_unknown_config_is_rejected():
    with pytest.raises(FileNotFoundError):
        resolve_config_path("does_not_exist")


def test_config_values_are_loaded():
    config = load_data_config(CONFIG_NAME)
    assert config.step_seconds == 21600
    assert config.source.stored_levels == 13
    assert config.precipitation_channels == ("total_precipitation_6hr",)
    assert config.cache_root == PROJECT_ROOT / "artifacts" / "data" / "era5_5p6deg"


def test_cache_root_can_be_replaced(tmp_path):
    config = load_data_config(CONFIG_NAME, cache_root=str(tmp_path))
    assert config.cache_root == Path(tmp_path)


def test_steps_follow_the_step_length():
    assert expected_steps(2010, 6 * 3600) == 1460
    assert expected_steps(2012, 6 * 3600) == 1464
    assert expected_steps(2010, 3 * 3600) == 2920


def test_plan_follows_the_step_length(tmp_path):
    six = make_config(tmp_path, step_hours=6)
    three = make_config(tmp_path, step_hours=3)
    assert steps_per_year(six) == 1461
    assert steps_per_year(three) == 2922
    assert read_bytes_per_year(three) == 2 * read_bytes_per_year(six)


def test_years_must_belong_to_the_splits(tmp_path):
    config = make_config(tmp_path)
    assert select_years(config) == [2010, 2011]
    assert select_years(config, [2011]) == [2011]
    with pytest.raises(ValueError):
        select_years(config, [1999])


def test_three_hourly_data_loads(tmp_path):
    config = make_config(tmp_path / "cache", step_hours=3)
    make_cache(config)
    build_statistics(config)
    assert check_cache(config) == []
    assert build_dataset(config, "val")[0]["frames"].shape == (5, 3, 8, 16)
