import numpy as np
from tqdm import tqdm


def open_source(source):
    import xarray

    return xarray.open_zarr(
        source.url,
        chunks=None,
        storage_options=source.storage_options,
    )


def close_source(dataset):
    dataset.close()
    try:
        from zarr.core.sync import cleanup_resources
    except ImportError:
        return
    cleanup_resources()


def select_year(dataset, year):
    return dataset.sel(time=slice(f"{year}-01-01", f"{year}-12-31T23:59:59"))


def read_pressure_variable(year_data, variable, levels):
    selected = year_data[variable].sel(level=list(levels))
    return selected.transpose("time", "level", "latitude", "longitude").values


def read_surface_variable(year_data, variable):
    ordered = year_data[variable].transpose("time", "latitude", "longitude")
    return ordered.values[:, None]


def stack_channels(year_data, channels, show_progress=False):
    jobs = [("pressure", name) for name in channels.pressure_variables]
    jobs += [("surface", name) for name in channels.surface_variables]
    arrays = []
    for kind, variable in tqdm(jobs, desc="variables", leave=False, disable=not show_progress):
        if kind == "pressure":
            arrays.append(read_pressure_variable(year_data, variable, channels.levels))
        else:
            arrays.append(read_surface_variable(year_data, variable))
    return np.concatenate(arrays, axis=1).astype(np.float32)


def year_seconds(year_data):
    return year_data["time"].values.astype("datetime64[s]").astype(np.int64)


def load_static(dataset, field_names):
    fields = {
        name: dataset[name].transpose("latitude", "longitude").values.astype(np.float32)
        for name in field_names
    }
    return {
        "latitudes": dataset["latitude"].values.astype(np.float64),
        "longitudes": dataset["longitude"].values.astype(np.float64),
        **fields,
    }
