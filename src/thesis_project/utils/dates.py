import re
import warnings
from datetime import date
from typing import Optional

_DATE_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})")


def filename_to_date(
    filename: str,
) -> Optional[date]:
    """
    Extract date from filename: <lob_type>_lob_freq_1s_<YYYY>_<MM>_<DD>.parquet

    Args:
        filename: The name of the file to extract the date from.

    Returns:
        A date object if the filename contains a valid date, otherwise None.
    """
    match = _DATE_PATTERN.search(filename)
    if not match:
        warnings.warn(f"Filename '{filename}' does not match expected date pattern.", UserWarning)
        return None
    year, month, day = map(int, match.groups())
    return date(year, month, day)
