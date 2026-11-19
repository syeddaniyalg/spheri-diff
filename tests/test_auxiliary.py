import numpy as np

from spheridiff.data.auxiliary import day_fraction, static_auxiliary, time_auxiliary, year_fraction


def seconds_of(text):
    return np.array([np.datetime64(text, "s").astype(np.int64)])


def test_year_fraction_at_start_and_middle():
    assert year_fraction(seconds_of("2010-01-01T00:00"))[0] == 0.0
    assert np.isclose(year_fraction(seconds_of("2010-07-02T12:00"))[0], 0.5, atol=0.01)


def test_day_fraction_follows_longitude():
    longitudes = np.array([0.0, 90.0, 180.0])
    fractions = day_fraction(seconds_of("2010-01-01T00:00"), longitudes)[0]
    assert np.allclose(fractions, [0.0, 0.25, 0.5])


def test_time_auxiliary_shape_and_range():
    seconds = seconds_of("2010-01-01T00:00") + 21600 * np.arange(5)
    maps = time_auxiliary(seconds, np.arange(16) * 22.5, 8)
    assert maps.shape == (5, 4, 8, 16)
    assert np.abs(maps).max() <= 1.0


def test_static_auxiliary_has_six_maps():
    static = {
        "latitudes": np.linspace(-90.0, 90.0, 8),
        "longitudes": np.arange(16) * 22.5,
        "land_sea_mask": np.ones((8, 16), dtype=np.float32),
        "soil_type": np.arange(128, dtype=np.float32).reshape(8, 16),
        "geopotential_at_surface": np.zeros((8, 16), dtype=np.float32),
    }
    maps = static_auxiliary(static, ["land_sea_mask", "soil_type", "geopotential_at_surface"])
    assert maps.shape == (6, 8, 16)
    assert np.allclose(maps[2], 0.0)
