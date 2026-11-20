from dataclasses import dataclass

INTERNAL_GRIDS = ("data", "gauss")
TRANSFORMS = ("dense", "separable")


@dataclass(frozen=True)
class SpherIDiffConfig:
    widths: tuple
    heads: int = 4
    rank: int = None
    internal_grid: str = "data"
    transform: str = "dense"

    @property
    def blocks(self):
        return len(self.widths) - 1

    def check_widths(self):
        if len(self.widths) < 2:
            raise ValueError("widths needs at least two entries, one per block boundary")
        for width in self.widths[1:]:
            if width % self.heads or (width // self.heads) % 2:
                raise ValueError(f"width {width} does not suit {self.heads} attention heads")

    def check_spectral_options(self):
        if self.internal_grid not in INTERNAL_GRIDS:
            raise ValueError(f"unknown internal grid '{self.internal_grid}', choose from {INTERNAL_GRIDS}")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unknown transform '{self.transform}', choose from {TRANSFORMS}")
        if self.transform == "separable" and self.internal_grid != "gauss":
            raise ValueError("the separable transform runs on the internal Gauss grid only")
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be a positive number")

    def __post_init__(self):
        self.check_widths()
        self.check_spectral_options()


def spheridiff_config_from_dict(raw):
    converted = {key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()}
    return SpherIDiffConfig(**converted)
