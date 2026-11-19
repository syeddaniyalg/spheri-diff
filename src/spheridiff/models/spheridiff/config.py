from dataclasses import dataclass


@dataclass(frozen=True)
class SpherIDiffConfig:
    widths: tuple
    heads: int = 4

    @property
    def blocks(self):
        return len(self.widths) - 1

    def __post_init__(self):
        if len(self.widths) < 2:
            raise ValueError("widths needs at least two entries, one per block boundary")
        for width in self.widths[1:]:
            if width % self.heads or (width // self.heads) % 2:
                raise ValueError(f"width {width} does not suit {self.heads} attention heads")


def spheridiff_config_from_dict(raw):
    converted = {key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()}
    return SpherIDiffConfig(**converted)
