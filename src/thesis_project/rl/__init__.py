from gymnasium.envs.registration import register

from . import dataset as dataset
from . import env_execution as env_execution
from . import env_trading as env_trading
from . import evaluation_execution as evaluation_execution
from . import evaluation_trading as evaluation_trading
from . import features as features
from . import policies as policies
from . import report as report
from . import train_execution as train_execution
from . import train_trading as train_trading

register(id="BasisTradingEnv-v0", entry_point="thesis_project.rl.env_trading:BasisTradingEnv")
register(id="BidExecutionEnv-v0", entry_point="thesis_project.rl.env_execution:BidExecutionEnv")
register(id="AskExecutionEnv-v0", entry_point="thesis_project.rl.env_execution:AskExecutionEnv")
