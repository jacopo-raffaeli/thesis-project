from gymnasium.envs.registration import register

from . import dataset as dataset
from . import env as env
from . import evaluation as evaluation
from . import features as features
from . import report as report
from . import train as train

register(id="BasisTradingEnv-v0", entry_point="thesis_project.rl.env:BasisTradingEnv")
