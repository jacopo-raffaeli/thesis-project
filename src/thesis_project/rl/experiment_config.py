from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    split_month: str

    total_timesteps: int = 1_000_000
    n_seeds: int = 5
    seed: int = 42

    batch_sizes: tuple[int, ...] = (64, 128, 256, 512)
    clip_ranges: tuple[float, ...] = (0.1, 0.2, 0.3)

    normalize_market_obs: bool = True

    def __post_init__(self):
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be > 0")

        if self.n_seeds <= 0:
            raise ValueError("n_seeds must be > 0")

        if not self.batch_sizes:
            raise ValueError("batch_sizes must not be empty")

        if not self.clip_ranges:
            raise ValueError("clip_ranges must not be empty")
