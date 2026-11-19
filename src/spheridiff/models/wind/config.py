from dataclasses import dataclass

BLOCK_KINDS = ("res", "transformer")


@dataclass(frozen=True)
class WindConfig:
    widths: tuple
    block_kinds: tuple
    blocks_per_level: tuple
    middle_blocks: int
    embedding_dim: int = 64
    heads: int = 4
    patch_size: int = 1
    norm_groups: int = 32
    model_height: int = None
    model_width: int = None

    @property
    def levels(self):
        return len(self.widths)

    def __post_init__(self):
        self.check_lengths()
        self.check_resampling()
        self.check_blocks()

    def check_lengths(self):
        if len(self.block_kinds) != self.levels:
            raise ValueError("block_kinds needs one entry per level")
        if len(self.blocks_per_level) != self.levels - 1:
            raise ValueError("blocks_per_level needs one entry per level except the last")
        unknown = set(self.block_kinds) - set(BLOCK_KINDS)
        if unknown:
            raise ValueError(f"unknown block kinds {sorted(unknown)}")

    def check_resampling(self):
        for upper, lower in zip(self.widths[:-1], self.widths[1:]):
            if lower % 4 or (4 * upper) % lower:
                raise ValueError(
                    f"widths {upper} and {lower} are not compatible with 2x2 pixel shuffling"
                )

    def check_blocks(self):
        for width, kind in zip(self.widths, self.block_kinds):
            if kind == "res" and width % self.norm_groups:
                raise ValueError(f"width {width} is not divisible by the {self.norm_groups} norm groups")
            if kind == "transformer" and not self.valid_attention_width(width):
                raise ValueError(f"width {width} does not suit {self.heads} heads")

    def valid_attention_width(self, width):
        return width % self.heads == 0 and (width // self.heads) % 2 == 0


def wind_config_from_dict(raw):
    converted = {key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()}
    return WindConfig(**converted)
