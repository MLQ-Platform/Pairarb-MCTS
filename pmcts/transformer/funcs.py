import numpy as np
import pandas as pd


def transform_spreads_to_signals(
    spreads: pd.DataFrame,
    enter_thr: float,
    close_thr: float,
) -> pd.DataFrame:
    """
    Transforms spread dataframe to signal dataframe.
    """
    # (N, num_pairs)
    signals = spreads.applymap(
        lambda x: 1 if abs(x) >= enter_thr else (-1 if abs(x) <= close_thr else 0)
    )
    return signals


def transform_spreads_to_positions(
    spreads: pd.DataFrame,
    enter_thr: float,
    close_thr: float,
    top_k: int = None,
) -> pd.DataFrame:
    """
    Transforms spread dataframe to position dataframe.
    """
    # (N, num_pairs)
    signals = transform_spreads_to_signals(
        spreads=spreads, enter_thr=enter_thr, close_thr=close_thr
    )
    # (N, num_pairs)
    positions = signals.replace(0, np.nan)
    positions = positions.ffill()
    positions = positions.replace(-1, 0)
    positions = positions.fillna(0)
    # (N, num_pairs)
    if top_k:
        positions = select_top_positions(positions, spreads, top_k)
    return positions


def select_top_positions(
    positions: pd.DataFrame, spreads: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    """
    Selects top k positions from positions dataframe.
    """
    top_positions = []

    # (num_pairs,)
    position_vector_old = np.zeros(positions.shape[1])

    for position_vector, spread_vector in zip(positions.values, spreads.values):
        numcut = int(position_vector.sum() - top_k)

        if numcut > 0:
            position_diff = position_vector - position_vector_old
            position_mask = np.where(position_diff <= 0, np.inf, position_diff)
            position_score = np.abs(spread_vector) * position_mask
            cutindex = np.argsort(position_score)[:numcut]
            position_vector[cutindex] = 0

        top_positions.append(position_vector)
        position_vector_old = position_vector

    # (N, num_pairs)
    top_positions = np.array(top_positions)
    top_positions = pd.DataFrame(
        top_positions, index=positions.index, columns=positions.columns
    )
    return top_positions
