import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from thesis_project import rl
from thesis_project.rl.train import make_serial_env


def evaluate_model(
    model: MaskablePPO,
    dataset: rl.env.RLDataset,
    env_config: rl.env.EnvConfig,
) -> pd.DataFrame:
    env = make_serial_env(dataset, env_config)

    records = []

    for episode in range(dataset.n_dates):
        obs, info = env.reset()

        while True:
            action_masks = get_action_masks(env)

            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks,
            )

            obs, _, terminated, truncated, info = env.step(int(action))

            record = info.copy()
            record["episode"] = episode
            records.append(record)

            if terminated or truncated:
                break

    env.close()

    return pd.DataFrame(records)
