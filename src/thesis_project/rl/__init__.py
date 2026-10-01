from gymnasium.envs.registration import register

from . import dataset as dataset
from . import env_trading as env_trading
from . import evaluation as evaluation
from . import features as features
from . import policies as policies
from . import report as report
from . import train as train

register(id="BasisTradingEnv-v0", entry_point="thesis_project.rl.env_trading:BasisTradingEnv")
