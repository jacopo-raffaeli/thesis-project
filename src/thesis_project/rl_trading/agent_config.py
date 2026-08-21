from dataclasses import dataclass


@dataclass(frozen=True)
class CommonAgentConfig:
    seed: int = 42
    device: str = "auto"
    verbose: int = 1
