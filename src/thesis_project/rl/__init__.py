from gymnasium.envs.registration import register

from . import dataset as dataset
from . import env as env
from . import features as features

register(id="BasisTradingEnv-v0", entry_point="thesis_project.rl.env:BasisTradingEnv")
