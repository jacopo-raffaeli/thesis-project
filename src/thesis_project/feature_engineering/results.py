"""
Results management for feature engineering analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def save_chunk(
    batch_results: List[Dict],
    chunk_dir: Path,
    chunk_number: int,
) -> int:
    """Save batch of results to parquet chunk."""
    if not batch_results:
        return chunk_number

    # Create directory
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    batch_df = pd.DataFrame(batch_results)

    # Verify required columns
    required_cols = ["horizon", "target", "feature", "n_samples", "timestamp"]
    missing = set(required_cols) - set(batch_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Save chunk
    chunk_path = chunk_dir / f"chunk_{chunk_number}.parquet"
    try:
        batch_df.to_parquet(chunk_path, index=False)
        logger.info(f"Saved chunk {chunk_number}: {chunk_path.name} ({len(batch_df)} rows)")
        return chunk_number + 1
    except Exception as e:
        raise IOError(f"Failed to save chunk {chunk_number} to {chunk_path}: {e}")


def load_and_concat_chunks(chunk_dir: Path) -> pd.DataFrame:
    """Load and concatenate all chunk files from directory."""
    chunk_files = sorted(chunk_dir.glob("chunk_*.parquet"))

    if not chunk_files:
        raise IOError(f"No chunk files found in {chunk_dir}")

    logger.info(f"Loading {len(chunk_files)} chunks from {chunk_dir}...")

    chunks = []
    for chunk_file in chunk_files:
        try:
            df = pd.read_parquet(chunk_file)
            chunks.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {chunk_file}: {e}")
            continue

    if not chunks:
        raise IOError(f"No chunks loaded from {chunk_dir}")

    result = pd.concat(chunks, ignore_index=True)
    logger.info(f"Concatenated {len(chunks)} chunks: {len(result)} total rows")

    return result
