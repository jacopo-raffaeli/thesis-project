# CONFIGURATION
DATA_DIR = "./data"
CASH_DIR = "/processed/daily"
FUTURE_DIR = "C://Users//u497575/Documents/IKA"
CTD_BONDS_FILE = "HCDT_IK1.csv"
MTS_PROPOSALS_FILE = "mts_proposals.parquet"
FUTURE_LOB_SUBDIR = "LOB"
OUTPUT_DIR = "./cache"

# LOB PARAMETERS
FREQ = "1s"
START_YEAR = 2022
END_YEAR = 2025
# Number of LOB levels
LEVELS = 10


def get_test_config():
    """
    Return test configuration as dictionary.
    """

    return {
        "data_dir": DATA_DIR,
        "cash_dir": CASH_DIR,
        "future_dir": FUTURE_DIR,
        "ctd_bonds_file": CTD_BONDS_FILE,
        "mts_proposals_file": MTS_PROPOSALS_FILE,
        "future_lob_subdir": FUTURE_LOB_SUBDIR,
        "output_dir": OUTPUT_DIR,
        "freq": FREQ,
        "levels": LEVELS,
    }
