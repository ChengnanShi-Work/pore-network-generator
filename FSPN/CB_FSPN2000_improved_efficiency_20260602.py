import numpy as np
import matplotlib.pyplot as plt
import openpnm as op
from pathlib import Path
from scipy import stats
from copulas.multivariate import GaussianMultivariate
import sobol_seq
import warnings
import time
import pandas as pd
import random
import sys
from scipy.spatial import cKDTree
from time import perf_counter
import os

warnings.filterwarnings("ignore")

S_pb = {}
S_pt = {}

# ----------------------------------------------------------S1----------------------------------------------------------
# ----------------------------------------------------------S1----------------------------------------------------------
path1 = Path(r'C:\2_Singlescale_CB\Extracted_from_15mm_4.8um_1')
project1 = op.io.Statoil.load(path=path1, prefix='CB_15mm_4.8um_1.raw')
pn1 = project1.network
# CN1 = pn1.num_neighbors(pores=pn1.Ps)

path2 = Path(r'C:\2_Singlescale_CB\Extracted_from_15mm_4.8um_2')
project2 = op.io.Statoil.load(path=path2, prefix='CB_15mm_4.8um_2_1600.raw')
pn2 = project2.network
# CN2 = pn2.num_neighbors(pores=pn2.Ps)

path3 = Path(r'C:\2_Singlescale_CB\Extracted_from_15mm_4.8um_3')
project3 = op.io.Statoil.load(path=path3, prefix='CB_15mm_4.8um_3_1600.raw')
pn3 = project3.network
# CN3 = pn3.num_neighbors(pores=pn3.Ps)

S1_b = pd.DataFrame({
    'pore.radius': np.concatenate([pn1['pore.radius'], pn2['pore.radius'], pn3['pore.radius']]),
    'pore.volume': np.concatenate([pn1['pore.volume'], pn2['pore.volume'], pn3['pore.volume']]),
    # 'CN': np.concatenate([CN1, CN2, CN3]),
    'pore.shape_factor': np.concatenate(
        [pn1['pore.shape_factor'], pn2['pore.shape_factor'], pn3['pore.shape_factor']])})
S1_t = pd.DataFrame({
    'throat.radius': np.concatenate([pn1['throat.radius'], pn2['throat.radius'], pn3['throat.radius']]),
    'throat.volume': np.concatenate([pn1['throat.volume'], pn2['throat.volume'], pn3['throat.volume']]),
    'throat.shape_factor': np.concatenate(
        [pn1['throat.shape_factor'], pn2['throat.shape_factor'], pn3['throat.shape_factor']])})

S_pb[14] = S1_b[(np.log10(S1_b['pore.radius']) >= -3.5) & (np.log10(S1_b['pore.radius']) < -3.25)]
S_pb[13] = S1_b[(np.log10(S1_b['pore.radius']) >= -3.75) & (np.log10(S1_b['pore.radius']) < -3.5)]
S_pb[12] = S1_b[(np.log10(S1_b['pore.radius']) >= -4) & (np.log10(S1_b['pore.radius']) < -3.75)]
S_pb[11] = S1_b[(np.log10(S1_b['pore.radius']) >= -4.25) & (np.log10(S1_b['pore.radius']) < -4)]
S_pb[10] = S1_b[(np.log10(S1_b['pore.radius']) >= -4.5) & (np.log10(S1_b['pore.radius']) < -4.25)]
S_pb[9] = S1_b[(np.log10(S1_b['pore.radius']) >= -4.75) & (np.log10(S1_b['pore.radius']) < -4.5)]

S_pt[14] = S1_t[(np.log10(S1_t['throat.radius']) >= -3.5) & (np.log10(S1_t['throat.radius']) < -3.25)]
S_pt[13] = S1_t[(np.log10(S1_t['throat.radius']) >= -3.75) & (np.log10(S1_t['throat.radius']) < -3.5)]
S_pt[12] = S1_t[(np.log10(S1_t['throat.radius']) >= -4) & (np.log10(S1_t['throat.radius']) < -3.75)]
S_pt[11] = S1_t[(np.log10(S1_t['throat.radius']) >= -4.25) & (np.log10(S1_t['throat.radius']) < -4)]
S_pt[10] = S1_t[(np.log10(S1_t['throat.radius']) >= -4.5) & (np.log10(S1_t['throat.radius']) < -4.25)]
S_pt[9] = S1_t[(np.log10(S1_t['throat.radius']) >= -4.75) & (np.log10(S1_t['throat.radius']) < -4.5)]

# ----------------------------------------------------------S2----------------------------------------------------------
path1 = Path(r'C:\2_Singlescale_CB\Extracted_from_15mm_2.4um_1')
project1 = op.io.Statoil.load(path=path1, prefix='CB_15mm_2.4um_1_1800.raw')
pn1 = project1.network
# CN1 = pn1.num_neighbors(pores=pn1.Ps)

path2 = Path(r'C:\2_Singlescale_CB\Extracted_from_15mm_2.4um_2')
project2 = op.io.Statoil.load(path=path2, prefix='CB_15mm_2.4um_2_1800.raw')
pn2 = project2.network
# CN2 = pn2.num_neighbors(pores=pn2.Ps)

path3 = Path(r'C:\2_Singlescale_CB\Extracted_from_15mm_2.4um_3')
project3 = op.io.Statoil.load(path=path3, prefix='CB_15mm_2.4um_3_1800.raw')
pn3 = project3.network
# CN3 = pn3.num_neighbors(pores=pn3.Ps)

S2_b = pd.DataFrame({
    'pore.radius': np.concatenate([pn1['pore.radius'], pn2['pore.radius'], pn3['pore.radius']]),
    'pore.volume': np.concatenate([pn1['pore.volume'], pn2['pore.volume'], pn3['pore.volume']]),
    # 'CN': np.concatenate([CN1, CN2, CN3]),
    'pore.shape_factor': np.concatenate(
        [pn1['pore.shape_factor'], pn2['pore.shape_factor'], pn3['pore.shape_factor']])})

S2_t = pd.DataFrame({
    'throat.radius': np.concatenate([pn1['throat.radius'], pn2['throat.radius'], pn3['throat.radius']]),
    'throat.volume': np.concatenate([pn1['throat.volume'], pn2['throat.volume'], pn3['throat.volume']]),
    'throat.shape_factor': np.concatenate(
        [pn1['throat.shape_factor'], pn2['throat.shape_factor'], pn3['throat.shape_factor']])})
S_pb[8] = S2_b[(np.log10(S2_b['pore.radius']) >= -5) & (np.log10(S2_b['pore.radius']) < -4.75)]
S_pb[7] = S2_b[(np.log10(S2_b['pore.radius']) >= -5.25) & (np.log10(S2_b['pore.radius']) < -5)]
S_pt[8] = S2_t[(np.log10(S2_t['throat.radius']) >= -5) & (np.log10(S2_t['throat.radius']) < -4.75)]

# ----------------------------------------------------------S4----------------------------------------------------------
path1 = Path(r'C:\2_Singlescale_CB\Extracted_from_7mm_1.2um_1')
project1 = op.io.Statoil.load(path=path1, prefix='CB_7mm_1.2um_1_1800.raw')
pn1 = project1.network
# CN1 = pn1.num_neighbors(pores=pn1.Ps)

path2 = Path(r'C:\2_Singlescale_CB\Extracted_from_7mm_1.2um_2')
project2 = op.io.Statoil.load(path=path2, prefix='CB_7mm_1.2um_2_1800.raw')
pn2 = project2.network
# CN2 = pn2.num_neighbors(pores=pn2.Ps)

path3 = Path(r'C:\2_Singlescale_CB\Extracted_from_7mm_1.2um_3')
project3 = op.io.Statoil.load(path=path3, prefix='CB_7mm_1.2um_3_1800.raw')
pn3 = project3.network
# CN3 = pn3.num_neighbors(pores=pn3.Ps)
# CN_max = max(CN1.max(), CN2.max(), CN3.max())
# print(CN_max)
S4_b = pd.DataFrame({
    'pore.radius': np.concatenate([pn1['pore.radius'], pn2['pore.radius'], pn3['pore.radius']]),
    'pore.volume': np.concatenate([pn1['pore.volume'], pn2['pore.volume'], pn3['pore.volume']]),
    # 'CN': np.concatenate([CN1, CN2, CN3]),
    'pore.shape_factor': np.concatenate(
        [pn1['pore.shape_factor'], pn2['pore.shape_factor'], pn3['pore.shape_factor']])})

S4_t = pd.DataFrame({
    'throat.radius': np.concatenate([pn1['throat.radius'], pn2['throat.radius'], pn3['throat.radius']]),
    'throat.volume': np.concatenate([pn1['throat.volume'], pn2['throat.volume'], pn3['throat.volume']]),
    'throat.shape_factor': np.concatenate(
        [pn1['throat.shape_factor'], pn2['throat.shape_factor'], pn3['throat.shape_factor']])})

S_pb[6] = S4_b[(np.log10(S4_b['pore.radius']) >= -5.5) & (np.log10(S4_b['pore.radius']) < -5.25)]
S_pt[7] = S4_t[(np.log10(S4_t['throat.radius']) >= -5.25) & (np.log10(S4_t['throat.radius']) < -5)]
S_pt[6] = S4_t[(np.log10(S4_t['throat.radius']) >= -5.5) & (np.log10(S4_t['throat.radius']) < -5.25)]
# ----------------------------------------------------------S5----------------------------------------------------------

path1 = Path(r'C:\2_Singlescale_CB\Extracted_from_2.5mm_0.6um_1')
project1 = op.io.Statoil.load(path=path1, prefix='CB_2.5mm_0.6um_1_1700.raw')
pn1 = project1.network
# CN1 = pn1.num_neighbors(pores=pn1.Ps)

path2 = Path(r'C:\2_Singlescale_CB\Extracted_from_2.5mm_0.6um_2')
project2 = op.io.Statoil.load(path=path2, prefix='CB_2.5mm_0.6um_2_1800.raw')
pn2 = project2.network
# CN2 = pn2.num_neighbors(pores=pn2.Ps)

path3 = Path(r'C:\2_Singlescale_CB\Extracted_from_2.5mm_0.6um_3')
project3 = op.io.Statoil.load(path=path3, prefix='CB_2.5mm_0.6um_3_1500.raw')
pn3 = project3.network
# CN3 = pn3.num_neighbors(pores=pn3.Ps)

S5_b = pd.DataFrame({
    'pore.radius': np.concatenate([pn1['pore.radius'], pn2['pore.radius'], pn3['pore.radius']]),
    'pore.volume': np.concatenate([pn1['pore.volume'], pn2['pore.volume'], pn3['pore.volume']]),
    # 'CN': np.concatenate([CN1, CN2, CN3]),
    'pore.shape_factor': np.concatenate(
        [pn1['pore.shape_factor'], pn2['pore.shape_factor'], pn3['pore.shape_factor']])})

S5_t = pd.DataFrame({
    'throat.radius': np.concatenate([pn1['throat.radius'], pn2['throat.radius'], pn3['throat.radius']]),
    'throat.volume': np.concatenate([pn1['throat.volume'], pn2['throat.volume'], pn3['throat.volume']]),
    'throat.shape_factor': np.concatenate(
        [pn1['throat.shape_factor'], pn2['throat.shape_factor'], pn3['throat.shape_factor']])})

S_pb[5] = S5_b[(np.log10(S5_b['pore.radius']) >= -5.75) & (np.log10(S5_b['pore.radius']) < -5.5)]
S_pt[5] = S5_t[(np.log10(S5_t['throat.radius']) >= -5.75) & (np.log10(S5_t['throat.radius']) < -5.5)]

# ----------------------------------------------------------S6----------------------------------------------------------
path1 = Path(r'C:\2_Singlescale_CB\Extracted_from_1.5mm_0.3um_1')
project1 = op.io.Statoil.load(path=path1, prefix='CB_1.5mm_0.3um_1_1800.raw')
pn1 = project1.network
# CN1 = pn1.num_neighbors(pores=pn1.Ps)

path2 = Path(r'C:\2_Singlescale_CB\Extracted_from_1.5mm_0.3um_2')
project2 = op.io.Statoil.load(path=path2, prefix='CB_1.5mm_0.3um_2_1900.raw')
pn2 = project2.network
# CN2 = pn2.num_neighbors(pores=pn2.Ps)

path3 = Path(r'C:\2_Singlescale_CB\Extracted_from_1.5mm_0.3um_3')
project3 = op.io.Statoil.load(path=path3, prefix='CB_1.5mm_0.3um_3_1800.raw')
pn3 = project3.network
# CN3 = pn3.num_neighbors(pores=pn3.Ps)

S6_b = pd.DataFrame({
    'pore.radius': np.concatenate([pn1['pore.radius'], pn2['pore.radius'], pn3['pore.radius']]),
    'pore.volume': np.concatenate([pn1['pore.volume'], pn2['pore.volume'], pn3['pore.volume']]),
    # 'CN': np.concatenate([CN1, CN2, CN3]),
    'pore.shape_factor': np.concatenate(
        [pn1['pore.shape_factor'], pn2['pore.shape_factor'], pn3['pore.shape_factor']])})

S6_t = pd.DataFrame({
    'throat.radius': np.concatenate([pn1['throat.radius'], pn2['throat.radius'], pn3['throat.radius']]),
    'throat.volume': np.concatenate([pn1['throat.volume'], pn2['throat.volume'], pn3['throat.volume']]),
    'throat.shape_factor': np.concatenate(
        [pn1['throat.shape_factor'], pn2['throat.shape_factor'], pn3['throat.shape_factor']])})
S_pb[4] = S6_b[(np.log10(S6_b['pore.radius']) >= -6) & (np.log10(S6_b['pore.radius']) < -5.75)]
S_pt[4] = S6_t[(np.log10(S6_t['throat.radius']) >= -6) & (np.log10(S6_t['throat.radius']) < -5.75)]

# ----------------------------------------------------------S7----------------------------------------------------------
path1 = Path(r'C:\2_Singlescale_CB\Extracted_from_XRM_128nm_1')
project1 = op.io.Statoil.load(path=path1, prefix='S7_1')
pn1 = project1.network
# CN1 = pn1.num_neighbors(pores=pn1.Ps)

path2 = Path(r'C:\2_Singlescale_CB\Extracted_from_XRM_128nm_2')
project2 = op.io.Statoil.load(path=path2, prefix='S2')
pn2 = project2.network
# CN2 = pn2.num_neighbors(pores=pn2.Ps)

path3 = Path(r'C:\2_Singlescale_CB\Extracted_from_XRM_128nm_3')
project3 = op.io.Statoil.load(path=path3, prefix='S7_3')
pn3 = project3.network
# CN3 = pn3.num_neighbors(pores=pn3.Ps)

S7_b = pd.DataFrame({
    'pore.radius': np.concatenate([pn1['pore.radius'], pn2['pore.radius'], pn3['pore.radius']]),
    'pore.volume': np.concatenate([pn1['pore.volume'], pn2['pore.volume'], pn3['pore.volume']]),
    # 'CN': np.concatenate([CN1, CN2, CN3]),
    'pore.shape_factor': np.concatenate(
        [pn1['pore.shape_factor'], pn2['pore.shape_factor'], pn3['pore.shape_factor']])})

S7_t = pd.DataFrame({
    'throat.radius': np.concatenate([pn1['throat.radius'], pn2['throat.radius'], pn3['throat.radius']]),
    'throat.volume': np.concatenate([pn1['throat.volume'], pn2['throat.volume'], pn3['throat.volume']]),
    'throat.shape_factor': np.concatenate(
        [pn1['throat.shape_factor'], pn2['throat.shape_factor'], pn3['throat.shape_factor']])})

S_pb[3] = S7_b[(np.log10(S7_b['pore.radius']) >= -6.25) & (np.log10(S7_b['pore.radius']) < -6)]
S_pb[2] = S7_b[(np.log10(S7_b['pore.radius']) >= -6.5) & (np.log10(S7_b['pore.radius']) < -6.25)]
S_pb[1] = S7_b[(np.log10(S7_b['pore.radius']) >= -6.75) & (np.log10(S7_b['pore.radius']) < -6.5)]
S_pb[0] = S7_b[(np.log10(S7_b['pore.radius']) >= -7) & (np.log10(S7_b['pore.radius']) < -6.75)]

S_pt[3] = S7_t[(np.log10(S7_t['throat.radius']) >= -6.25) & (np.log10(S7_t['throat.radius']) < -6)]
S_pt[2] = S7_t[(np.log10(S7_t['throat.radius']) >= -6.5) & (np.log10(S7_t['throat.radius']) < -6.25)]
S_pt[1] = S7_t[(np.log10(S7_t['throat.radius']) >= -6.75) & (np.log10(S7_t['throat.radius']) < -6.5)]
S_pt[0] = S7_t[(np.log10(S7_t['throat.radius']) >= -7) & (np.log10(S7_t['throat.radius']) < -6.75)]


def valid_mask(df, ss):
    return (
        df['pore.volume'].between(ss['min_pv'], ss['max_pv']) &
        df['pore.radius'].between(ss['min_pr'], ss['max_pr']) &
        # df['CN'].between(0, ss['max_pCN']) &
        df['pore.shape_factor'].between(ss['min_ps'], 1/(4*np.pi))
    )


def valid_mask_pt(df, ss):
    return (
        df['throat.volume'].between(ss['min_tv'], ss['max_tv']) &
        df['throat.radius'].between(ss['min_tr'], ss['max_tr']) &
        df['throat.shape_factor'].between(ss['min_ts'], 1/(4*np.pi))
    )


def random_selection(pool, target, tolerance=0.01, seed=None,
                     attempts=50, max_first_fraction=0.70):
    rng = np.random.default_rng(seed)
    vol_pores = pool['pore.volume'].to_numpy()
    n = len(pool)

    tolerance_passes = [tolerance, 0.03, 0.05]

    best_picked = None
    best_total = 0.0
    best_error = np.inf

    for _ in range(attempts):
        indices = rng.permutation(n)

        picked = np.zeros(n, dtype=bool)
        total = 0.0

        for tol in tolerance_passes:
            lower = target * (1 - tol)
            upper = target * (1 + tol)

            remaining = indices[~picked[indices]]

            for a in remaining:
                v = vol_pores[a]

                if total == 0 and v >= max_first_fraction * target:
                    if lower <= v <= upper:
                        picked[a] = True
                        total = v
                        break
                    continue

                if total + v <= upper:
                    picked[a] = True
                    total += v

                if total >= lower:
                    break

            error = abs(total - target)
            if error < best_error:
                best_error = error
                best_picked = picked.copy()
                best_total = total

            if lower <= total <= upper:
                selected_idx = np.flatnonzero(picked)
                return pool.iloc[selected_idx], total

    print(
        f"No valid random selection found after {attempts} attempts. "
        f"Best total={best_total:.6E}, target={target:.6E}, "
        f"rel_error={(best_total - target) / target:.4f}"
    )

    return None, None


def random_selection_pt(pool, target, tolerance=0.1, seed=None):
    lower = target * (1 - tolerance)
    upper = target * (1 + tolerance)

    vol_throats = pool['throat.volume'].to_numpy()

    n = len(pool)

    indices_pt = sorted(range(n), key=lambda i: vol_throats[i], reverse=True)
    shuffle_part = n
    top_subset = indices_pt[:shuffle_part]
    np.random.shuffle(top_subset)
    indices_pt[:shuffle_part] = top_subset

    picked = [False] * n
    total = 0

    # 1) Random‐order greedy pass
    for a in indices_pt:
        v = vol_throats[a]
        if total + v <= upper:
            picked[a] = True
            total += v
        if total >= lower:
            break

    # 2) If still below lower, try filling with remaining in random order
    if total < lower:
        print(total, lower)
        tolerance = 0.03
        lower = target * (1 - tolerance)
        upper = target * (1 + tolerance)
        remaining = [a for a in indices_pt if not picked[a]]
        for a in remaining:
            v = vol_throats[a]
            if total + v <= upper:
                picked[a] = True
                total += v
            if total >= lower:
                break

    # 3) If still below lower, try filling with remaining in random order
    if total < lower:
        print(total, lower)
        tolerance = 0.05
        lower = target * (1 - tolerance)
        upper = target * (1 + tolerance)
        remaining = [a for a in indices_pt if not picked[a]]
        for a in remaining:
            v = vol_throats[a]
            if total + v <= upper:
                picked[a] = True
                total += v
            if total >= lower:
                break

    selected_idx = [a for a, flag_pt in enumerate(picked) if flag_pt]
    selected = pool.iloc[selected_idx]
    if lower <= total <= upper:
        return selected, total
    else:
        return None, None


def find_closest_valid_neighbor(
    pore1, throat_r, attempt=50,
    *,                   # force keyword-only for the extras
    valid_candidates=None,   # 1D int array from the current band
    weights=None,            # global weights array (same shape as pores)
    rng=None,                # pass the RNG you already created
    use_set=False            # keep set(link_matrix) check only if you must
):
    """
    Returns (pore2, p1_use) or (None, None).
    Faster because:
      - skips huge set membership unless use_set=True
      - resamples p1 via cumulative weights (no probs allocation)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Local refs (avoid global lookups in the loop)
    nv   = neighbor_valid
    nei  = neighbor_indices
    pr   = pore_radii
    CNv  = PoreBody[:, 6]  # view

    # Precompute cumulative weights for p1 re-sampling (if provided)
    if valid_candidates is not None and weights is not None and valid_candidates.size > 0:
        w = weights[valid_candidates]
        W = w.sum()
        cw = np.cumsum(w, dtype=np.float64) if W > 0 else None
    else:
        cw = None
        W  = 0.0

    p = int(pore1)
    tr = float(throat_r)

    for _ in range(int(attempt)):
        # scan currently-valid neighbor positions of p
        valid_pos = np.flatnonzero(nv[p])
        for pos in valid_pos:
            q = int(nei[p][pos])
            if pr[q] >= tr and CNv[q] <= CN_limit:
                if use_set:
                    conn = (min(p, q), max(p, q))
                    if conn in link_matrix:
                        continue
                # consume this neighbor position so we don't reuse it
                nv[p][pos] = False
                return q, p

        # Need a new p1 → resample from the current band without building probs
        if cw is not None and W > 0.0:
            u = rng.random() * W
            idx = np.searchsorted(cw, u, side="left")
            p = int(valid_candidates[idx])
        else:
            # last-resort fallback (uniform if no weights available)
            if valid_candidates is not None and valid_candidates.size > 0:
                p = int(valid_candidates[rng.integers(0, valid_candidates.size)])
            else:
                return None, None

    return None, None


def find_closest_valid_neighbor_fast(
    pore1,
    throat_r,
    attempt=50,
    *,
    valid_candidates,
    candidate_cw,
    candidate_weight_sum,
    rng,
    use_set=True
):
    nv = neighbor_valid
    nei = neighbor_indices
    pr = pore_radii
    CNv = PoreBody[:, 6]

    p = int(pore1)
    tr = float(throat_r)

    for _ in range(attempt):
        valid_pos = np.flatnonzero(nv[p])

        for pos in valid_pos:
            q = int(nei[p][pos])

            if pr[q] < tr or CNv[q] > CN_limit:
                nv[p][pos] = False
                continue

            if use_set:
                conn = (p, q) if p < q else (q, p)
                if conn in link_matrix:
                    nv[p][pos] = False
                    continue

            nv[p][pos] = False
            return q, p

        if candidate_weight_sum <= 0:
            return None, None

        idx = np.searchsorted(
            candidate_cw,
            rng.random() * candidate_weight_sum,
            side="left"
        )
        p = int(valid_candidates[idx])

    return None, None


def update_neighbors(idx):
    r_i = pore_radii[idx]
    log_r = np.log10(r_i)
    if log_r > -3.5:
        search_radius = 3 * r_i
    elif -4.5 < log_r <= -3.5:
        search_radius = 5 * r_i
    elif -5 < log_r <= -4.5:
        search_radius = 10 * r_i
    elif -6 < log_r <= -5:
        search_radius = 15 * r_i
    elif -6.5 < log_r <= -6:
        search_radius = 20 * r_i
    elif -7 < log_r <= -6.5:
        search_radius = 60 * r_i
    else:
        search_radius = 100 * r_i

    neighbors = tree.query_ball_point(pore_coord[idx], search_radius)
    neighbors = [j for j in neighbors if j != idx]
    return np.array(neighbors, dtype=int)


def print_copula_marginals(label, bin_id, columns, copula):
    print(f"\n[{label} bin {bin_id}] fitted marginal distributions:")
    for col, uni in zip(columns, copula.univariates):
        info = uni.to_dict()

        dist_type = info.get("type", type(uni).__name__)
        dist_type = dist_type.split(".")[-1]

        print(f"  {col}: {dist_type}")

        for key in ("loc", "scale", "a", "b", "mean", "std", "mu", "sigma"):
            if key in info:
                print(f"    {key}: {info[key]}")


def copula_has_kde(copula):
    for uni in copula.univariates:
        dist_type = uni.to_dict().get("type", "")
        if "GaussianKDE" in dist_type:
            return True
    return False


Sobol = True  # If you want to use random position of pores, then set sobol as False
T_in = time.time()
dim_new = [2000, 2000, 2000]  # Arbitrary dimensions defined by user
res = 1e-6
vol_new = dim_new[0] * dim_new[1] * dim_new[2] * (res ** 3)  # Volume of the stochastic network(m^3)
Lx = dim_new[0] * res  # Length of the X dimension in meters
Ly = dim_new[1] * res  # Length of the Y dimension in meters
Lz = dim_new[2] * res  # Length of the Z dimension in meters
bins = np.arange(-7, -3.25, 0.25)


target_porosity_bin = [0, 3.33271E-05, 0.000472586, 0.005611013, 0.00437332, 0.01090391, 0.01681518, 0.01392899,
                       0.01912526, 0.01468036, 0.01492312, 0.01528452, 0.0098973, 0.00541939, 0.00264339]
target_porosity_bin_pt = [5.90567E-05, 0.001085048, 0.012798496, 0.046484045, 0.009167901, 0.010347118, 0.0301673, 0.02440166,
                          0.02295626, 0.01098065, 0.00654153, 0.01055807, 0.00465245, 0.00383642, 0]


vol_list = [4.224E-5 ** 2 * 6.528E-5 * 2 + 4.224E-5 * 4.224E-5 * 5.76E-5, 5.4e-4 ** 3 * 2 + 5.7e-4 ** 3,
            1.02e-3 ** 3 + 1.08e-3 ** 3 + 9e-4 ** 3, 2.16e-3 ** 3 * 3, 4.32e-3 ** 3 * 3,
            8.64e-3 ** 3 + 7.68e-3 ** 3 * 2]
stats_pb = {}
stats_pt = {}

pore_copulas = {}
throat_copulas = {}
fitting_start = perf_counter()
pore_samplers = {}
throat_samplers = {}
for i in range(len(bins) - 2, 0, -1):
    df_i = S_pb[i]
    stats_pb[i] = {
        'max_pv': df_i['pore.volume'].max(),
        'min_pv': df_i['pore.volume'].min(),
        'max_pr': df_i['pore.radius'].max(),
        'min_pr': df_i['pore.radius'].min(),
        'min_ps': df_i['pore.shape_factor'].min()
    }

    if len(df_i) > 1000:
        df_fit = df_i.sample(n=1000, random_state=42).reset_index(drop=True)
    else:
        df_fit = df_i.reset_index(drop=True)

    fit_start = perf_counter()
    copula = GaussianMultivariate()
    copula.fit(df_fit)
    fit_end = perf_counter()

    pore_copulas[i] = copula
    print(f"[pore bin {i}] copula fit time: {fit_end - fit_start:.3f}s")
    print_copula_marginals("pore", i, df_fit.columns, copula)

    if copula_has_kde(copula):
        print(f"[pore bin {i}] KDE detected -> using raw-data sampling")
        pore_samplers[i] = {
            "mode": "raw",
            "data": df_i.reset_index(drop=True)
        }
    else:
        pore_samplers[i] = {
            "mode": "copula",
            "copula": copula
        }

for i in range(len(bins) - 3, 0, -1):
    df_i = S_pt[i]
    stats_pt[i] = {
        'max_tv': df_i['throat.volume'].max(),
        'min_tv': df_i['throat.volume'].min(),
        'max_tr': df_i['throat.radius'].max(),
        'min_tr': df_i['throat.radius'].min(),
        'min_ts': df_i['throat.shape_factor'].min()
    }

    if len(S_pt[i]) > 1000:
        df_fit = df_i.sample(n=1000, random_state=42).reset_index(drop=True)
    else:
        df_fit = df_i.reset_index(drop=True)

    fit_start = perf_counter()
    copula = GaussianMultivariate()
    copula.fit(df_fit)
    fit_end = perf_counter()

    throat_copulas[i] = copula
    print(f"[throat bin {i}] copula fit time: {fit_end - fit_start:.3f}s")
    print_copula_marginals("throat", i, df_fit.columns, copula)

    if copula_has_kde(copula):
        print(f"[throat bin {i}] KDE detected -> using raw-data sampling")
        throat_samplers[i] = {
            "mode": "raw",
            "data": df_i.reset_index(drop=True)
        }
    else:
        throat_samplers[i] = {
            "mode": "copula",
            "copula": copula
        }

print(f"Total fitting time: {perf_counter() - fitting_start:.3f}s")

Test = 1
Porosity = np.zeros([Test, 1])
for T in range(Test):
    start1 = time.time()
    total_pore_num = 0
    body = []
    throat = []

    for i in range(len(bins) - 2, 0, -1):

        start_fit = time.time()
        pore_body = []
        if i == 14 or i == 13 or i == 12 or i == 11 or i == 10 or i == 9:
            vol_org = vol_list[5]
        elif i == 8 or i == 7:
            vol_org = vol_list[4]
        elif i == 6:
            vol_org = vol_list[3]
        elif i == 5:
            vol_org = vol_list[2]
        elif i == 4:
            vol_org = vol_list[1]
        else:
            vol_org = vol_list[0]

        ori_pore_num = len(S_pb[i]['pore.radius'])
        pore_num = np.int_(np.round(ori_pore_num / vol_org) * vol_new)

        copula = pore_copulas[i]
        # print('copula fitting time: ', end_fit)

        start_sample = time.time()
        pool_size = max(pore_num * 2, pore_num + 1000)

        #print(pool_size)
        t0 = perf_counter()
        sampler = pore_samplers[i]
        if sampler["mode"] == "copula":
            pores = sampler["copula"].sample(pool_size)
        else:
            pores = sampler["data"].sample(
                n=pool_size,
                replace=True
            ).reset_index(drop=True)

        t1 = perf_counter()
        mask = valid_mask(pores, stats_pb[i])
        t2 = perf_counter()
        pores = pores[mask].reset_index(drop=True)

        print(f"[bin {i}] sample: {t1-t0:.4f}s | mask: {t2-t1:.4f}s | kept: {mask.sum()}/{pool_size}")
        end_sample = time.time()

        start_match = time.time()
        tol = 0.01
        target_volume = target_porosity_bin[i] * vol_new

        pore_body, current_volume = random_selection(pores, target_volume, tol)

        current_porosity = current_volume / vol_new
        diff = (current_porosity - target_porosity_bin[i]) / target_porosity_bin[i]

        total_pore_num += len(pore_body)
        body.append(pore_body)
        end_match = time.time() - end_sample
        print('matching time : ', end_match, 'target for', bins[i], 'is :', target_porosity_bin[i],
              ', current sub-porosity of ', len(pore_body), ' is :', current_porosity, ', difference: ', diff)

    Body = pd.concat(body, ignore_index=True)
    end1 = time.time() - start1
    print('Time for total body sampling is: ', end1, ', there are total ', total_pore_num, 'pore bodies.')

    start_distance = time.time()
    pore_coordRand = np.random.rand(total_pore_num, 3) * [Lx, Ly, Lz]  # Generate random position for pores
    sobolset = sobol_seq.i4_sobol_generate(3, 2 * total_pore_num)
    pore_coordSobol = sobolset[0:total_pore_num, :] * [Lx, Ly, Lz]  # Generate random Sobol position for pores
    if Sobol:
        pore_coord = pore_coordSobol
    else:
        pore_coord = pore_coordRand

    shuffled_indices = list(range(total_pore_num))
    random.shuffle(shuffled_indices)
    pore_coord = pore_coord[shuffled_indices]

    tree = cKDTree(pore_coord)
    pore_radii = Body['pore.radius'].values

    neighbor_indices = []
    for i, (p_c, r_i) in enumerate(zip(pore_coord, pore_radii)):
        log_r = np.log10(r_i)
        if log_r > -4:
            search_radius = 3 * r_i
        elif -4.5 < log_r <= -4:
            search_radius = 5 * r_i
        elif -5 < log_r <= -4.5:
            search_radius = 10 * r_i
        elif -6 < log_r <= -5:
            search_radius = 15 * r_i
        elif -6.5 < log_r <= -6:
            search_radius = 20 * r_i
        elif -7 < log_r <= -6.5:
            search_radius = 60 * r_i
        else:
            search_radius = 100 * r_i

        candi_nei = tree.query_ball_point(p_c, search_radius)
        candi_nei = [j for j in candi_nei if j != i]
        candi_nei = np.array(candi_nei, dtype=int)
        neighbor_indices.append(candi_nei)

    end_distance = time.time() - start_distance
    print('Distance calculation time is: ', end_distance)

    PoreBody = np.zeros([total_pore_num, 8])
    PoreBody[:, 0] = list(range(total_pore_num))
    PoreBody[:, 1:4] = pore_coord
    PoreBody[:, 4] = Body['pore.volume'].values
    PoreBody[:, 5] = Body['pore.radius'].values
    PoreBody[:, 7] = Body['pore.shape_factor'].values

    # ------------------------------------------------------------------------------------
    total_throat_num = 0
    start2 = time.time()

    for i in range(len(bins) - 3, 0, -1):

        start_fit = time.time()
        pore_throat = []
        if i == 14 or i == 13 or i == 12 or i == 11 or i == 10 or i == 9:
            vol_org = vol_list[5]
        elif i == 8:
            vol_org = vol_list[4]
        elif i == 7 or i == 6:
            vol_org = vol_list[3]
        elif i == 5:
            vol_org = vol_list[2]
        elif i == 4:
            vol_org = vol_list[1]
        else:
            vol_org = vol_list[0]

        ori_throat_num = len(S_pt[i]['throat.radius'])
        throat_num = np.int_(np.round(ori_throat_num / vol_org) * vol_new)

        copula = throat_copulas[i]
        end_fit = time.time() - start_fit
        # print('copula fitting time: ', end_fit)

        start_sample = time.time()
        sampler = throat_samplers[i]

        if i > 3:
            pool_size = max(throat_num * 2, throat_num + 1000)
            if sampler["mode"] == "copula":
                throats = sampler["copula"].sample(pool_size)
            else:
                throats = sampler["data"].sample(
                    n=pool_size,
                    replace=True
                ).reset_index(drop=True)
            throats = throats[valid_mask_pt(throats, stats_pt[i])].reset_index(drop=True)
            print(f"[bin {i}] requested {throat_num} → candidates after filter: {len(throats)}")
            end_sample = time.time() - start_sample
        else:
            pool_size = int(throat_num * 1.2)
            n1 = int(pool_size // 3)
            n2 = int(pool_size // 3)
            n3 = int(pool_size - n1 - n2)

            if sampler["mode"] == "copula":
                throats1 = sampler["copula"].sample(n1)
            else:
                throats1 = sampler["data"].sample(
                    n=n1,
                    replace=True
                ).reset_index(drop=True)
            throats1 = throats1[valid_mask_pt(throats1, stats_pt[i])].reset_index(drop=True)

            if sampler["mode"] == "copula":
                throats2 = sampler["copula"].sample(n2)
            else:
                throats2 = sampler["data"].sample(
                    n=n2,
                    replace=True
                ).reset_index(drop=True)
            throats2 = throats2[valid_mask_pt(throats2, stats_pt[i])].reset_index(drop=True)

            if sampler["mode"] == "copula":
                throats3 = sampler["copula"].sample(n3)
            else:
                throats3 = sampler["data"].sample(
                    n=n3,
                    replace=True
                ).reset_index(drop=True)
            throats3 = throats3[valid_mask_pt(throats3, stats_pt[i])].reset_index(drop=True)

            throats = pd.concat([throats1, throats2, throats3], ignore_index=True)
            print(f"[bin {i}] requested {throat_num} → candidates after filter: {len(throats)}")
            end_sample = time.time() - start_sample
            print('copula sampling time for', len(throats), ' pore throats : ', end_sample)

        start_match = time.time()
        tol = 0.01
        target_volume_pt = target_porosity_bin_pt[i] * vol_new

        pore_throat, current_volume_pt = random_selection_pt(throats, target_volume_pt, tol)

        current_porosity_pt = current_volume_pt / vol_new
        diff = (current_porosity_pt - target_porosity_bin_pt[i]) / target_porosity_bin_pt[i]
        total_throat_num += len(pore_throat)
        throat.append(pore_throat)
        end_match = time.time() - start_fit
        print('Time for pore throat sampling is: ', end_match, 'target for', bins[i], 'is :', target_porosity_bin_pt[i],
              'current sub-porosity of', len(pore_throat), ' is :', current_porosity_pt, 'difference: ', diff)

    Throat = pd.concat(throat, ignore_index=True)
    Throat = Throat.apply(pd.to_numeric, errors='coerce')
    end2 = time.time() - start2
    print('Time for total pore throat sampling is: ', end2, ', there are total ', total_throat_num, 'pore throats')

    # --------*********   Finding connected pores based on each pore coordination number
    start3 = time.time()
    start_time = start3
    sorted_Throat = Throat.sort_values(by="throat.radius", ascending=False)
    sorted_Throat = sorted_Throat.reset_index(drop=True)
    ThroatBody = np.zeros([total_throat_num, 6])
    ThroatBody[:, 3] = sorted_Throat['throat.volume']
    ThroatBody[:, 4] = sorted_Throat['throat.radius']
    ThroatBody[:, 5] = sorted_Throat['throat.shape_factor']

    link_matrix = set()
    rows_to_delete = []
    neighbor_valid = [np.ones(len(sublist), dtype=bool) for sublist in neighbor_indices]

    CN_limit = 3000
    alpha = 0.5
    eps = 1e-12
    weights = (pore_radii - pore_radii.min() + eps) ** alpha

    order_p = np.argsort(pore_radii)
    r_sorted = pore_radii[order_p].astype(np.float32)

    thr = ThroatBody[:, 4].astype(np.float32)  # already sorted desc
    cutoffs = np.searchsorted(r_sorted, thr, side='left')

    CN = PoreBody[:, 6]
    PR = PoreBody[:, 5]

    rng = np.random.default_rng()
    t_find = 0.0
    t_fallback = 0.0
    t_update = 0.0

    i = 0
    while i < total_throat_num:
        throat_radius = ThroatBody[i, 4]
        cutoff = int(cutoffs[i])

        j = i + 1
        while j < total_throat_num and cutoffs[j] == cutoff:
            j += 1
        m = j - i
        if m <= 0:
            i = j
            continue

        candidates_view = order_p[cutoff:]  # pores with r >= current throat radius
        if candidates_view.size == 0:
            rows_to_delete.extend(range(i, j))
            i = j
            continue

        # Filter CN in this window once
        cn_ok = (CN[candidates_view] <= CN_limit)
        valid_candidates = candidates_view[cn_ok]
        if valid_candidates.size < 2:
            rows_to_delete.extend(range(i, j))
            i = j
            continue

        # Build weights/cumweights ONCE for the whole batch
        w = weights[valid_candidates]
        W = w.sum()
        if W <= 0:
            rows_to_delete.extend(range(i, j))
            i = j
            continue
        cw = np.cumsum(w, dtype=np.float64)

        # Vectorized draw of m p1's (no per-throat probs/choice)
        u = rng.random(m) * W
        idx_in_valid = np.searchsorted(cw, u, side="left")
        p1_batch = valid_candidates[idx_in_valid].astype(int)

        # Now process the m throats (neighbor search untouched)
        for off, ii in enumerate(range(i, j)):
            throat_radius = ThroatBody[ii, 4]
            p1 = int(p1_batch[off])

            if CN[p1] > CN_limit:
                rows_to_delete.append(ii)
                continue
            """
            p2, p1_use = find_closest_valid_neighbor(
                p1, throat_radius, attempt=50,
                valid_candidates=valid_candidates,
                weights=weights,
                rng=rng,  # the RNG you created outside the loop
                use_set=True  # set to True only if you must enforce uniqueness here
            )
            """
            t0 = perf_counter()
            p2, p1_use = find_closest_valid_neighbor_fast(
                p1,
                throat_radius,
                attempt=50,
                valid_candidates=valid_candidates,
                candidate_cw=cw,
                candidate_weight_sum=W,
                rng=rng,
                use_set=True
            )
            t_find += perf_counter() - t0

            if p2 is None:
                # --- your original fallback, unchanged except it reuses window arrays ---
                # reweight within valid_candidates (cheap re-use)
                t0 = perf_counter()
                w = weights[valid_candidates]
                W = w.sum()
                if W <= 0:
                    rows_to_delete.append(ii)
                    continue
                cw = np.cumsum(w, dtype=np.float64)
                p1_retry = int(valid_candidates[np.searchsorted(cw, rng.random() * W, side="left")])
                p1_neighbors = neighbor_indices[p1_retry]
                if len(p1_neighbors) == 0:
                    rows_to_delete.append(ii)
                    continue
                ref_pore = int(np.random.choice(p1_neighbors))

                # “unlinked” via CN==0 (fast proxy); fall back to all if none
                unlinked_mask = (CN[valid_candidates] == 0)
                if np.any(unlinked_mask):
                    unlinked_pores = valid_candidates[unlinked_mask]
                else:
                    unlinked_pores = valid_candidates
                if len(unlinked_pores) == 0:
                    rows_to_delete.append(ii)
                    continue
                new_p2 = int(np.random.choice(unlinked_pores))

                # swap props (same as your code)
                for idx in (4, 5, 7):
                    PoreBody[new_p2, idx], PoreBody[ref_pore, idx] = PoreBody[ref_pore, idx], PoreBody[new_p2, idx]
                pore_radii[new_p2], pore_radii[ref_pore] = pore_radii[ref_pore], pore_radii[new_p2]

                # refresh neighbors/valid flags (your functions)
                neighbor_indices[ref_pore] = update_neighbors(ref_pore)
                neighbor_indices[new_p2] = update_neighbors(new_p2)
                neighbor_valid[ref_pore] = np.ones(len(neighbor_indices[ref_pore]), dtype=bool)
                neighbor_valid[new_p2] = np.ones(len(neighbor_indices[new_p2]), dtype=bool)

                # weights changed due to radii swap
                weights = (pore_radii - pore_radii.min() + eps) ** alpha

                p1_use = p1_retry
                p2 = ref_pore

                t_fallback += perf_counter() - t0

            t0 = perf_counter()
            # write connection
            ThroatBody[ii, 0] = p1_use
            ThroatBody[ii, 1] = p2
            # (we’ll vectorize column 2 after the loop; or leave this line if you prefer)
            # ThroatBody[ii, 2] = min(PR[p1_use], PR[p2])

            # update CN/weights incrementally
            CN[p1_use] += 1
            CN[p2] += 1
            if CN[p1_use] >= CN_limit: weights[p1_use] = 0.0
            if CN[p2] >= CN_limit:  weights[p2] = 0.0

            # keep your link_matrix if you must; consider dedup-at-end later for speed
            link_matrix.add((min(p1_use, p2), max(p1_use, p2)))
            t_update += perf_counter() - t0

        # progress (throttle)
        if i % 10000 == 0 or j == total_throat_num:
            elapsed = time.time() - start3
            sys.stdout.write(f"\rConnected throats: {i}/{total_throat_num} — elapsed: {elapsed:.1f}s")
            start3 = time.time()

        i = j  # next band

    print(f"Connection neighbor search time: {t_find:.3f}s")
    print(f"Connection fallback time: {t_fallback:.3f}s")
    print(f"Connection update/write time: {t_update:.3f}s")
    connected = (ThroatBody[:, 0] >= 0) & (ThroatBody[:, 1] >= 0)
    p1_ix = ThroatBody[connected, 0].astype(int)
    p2_ix = ThroatBody[connected, 1].astype(int)
    ThroatBody[connected, 2] = np.minimum(PR[p1_ix], PR[p2_ix])

    ThroatBody = np.delete(ThroatBody, rows_to_delete, axis=0)

    end3 = time.time() - start_time
    print('Time for connection is : ', end3, ', there are total', len(link_matrix), 'pore throats')

    # --------------------------------------Calculate porosity--------------------------------
    porosity = np.sum(target_porosity_bin) + np.sum(target_porosity_bin_pt)
    Porosity[T] = (np.sum(ThroatBody[:, 3]) + np.sum(PoreBody[:, 4])) / (Lx * Ly * Lz)
    # Calculate porosity of generated pore network
    print('Porosity is :', porosity)
    print('Original Porosity[', T + 1, '] of the stochastic network is : ', Porosity[T])
    diff = (Porosity[T] - porosity) / porosity  # Calculate difference
    print('Original Porosity[', T + 1, '] relatively difference is : ', diff)

    # ----------------------------Define Inlet & Outlet Pores-----------------------------------------
    # Step 1: Log-binning of throat radii
    throat_radii_log = np.log10(ThroatBody[:, 4])
    bin_edges = np.arange(np.floor(throat_radii_log.min()), np.ceil(throat_radii_log.max()) + 0.25, 0.25)
    bin_indices = np.digitize(throat_radii_log, bin_edges) - 1  # -1 to make it 0-based

    # Step 2: Create bin-wise statistics
    num_bins = len(bin_edges) - 1
    bin_avg_radius = np.zeros(num_bins)
    bin_avg_volume = np.zeros(num_bins)
    bin_avg_shape = np.zeros(num_bins)

    for i in range(num_bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_avg_radius[i] = np.mean(ThroatBody[bin_mask, 4])
            bin_avg_volume[i] = np.mean(ThroatBody[:, 3])
            bin_avg_shape[i] = np.mean(ThroatBody[bin_mask, 5])
        else:
            # Set to overall average if bin is empty
            bin_avg_radius[i] = np.mean(ThroatBody[:, 4])
            bin_avg_volume[i] = np.mean(ThroatBody[:, 3])
            bin_avg_shape[i] = np.mean(ThroatBody[:, 5])

    # Only one dimension is considered (x-coordination)
    # Define inlet pores
    inletIndex = np.zeros([len(PoreBody), 1])
    outletIndex = np.zeros([len(PoreBody), 1])
    for i in range(len(PoreBody)):
        x, r = PoreBody[i, 1], PoreBody[i, 5]

        # Determine bin of current pore radius
        log_r = np.log10(r)
        bin_idx = np.digitize(log_r, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)  # avoid index error

        # Assign inlet if close to x = 0
        if x - r < r:
            inletIndex[i, 0] = 1
            PoreBody[i, 6] += 1
            ThroatBody = np.r_[ThroatBody,
                               [[-2, i, 0, bin_avg_volume[bin_idx], bin_avg_radius[bin_idx], bin_avg_shape[bin_idx]]]]
            if r > 1E-4:
                print(i, bin_avg_volume[bin_idx], bin_avg_radius[bin_idx], bin_avg_shape[bin_idx])

            # Assign outlet if close to x = Lx
        if x + r > Lx - r:
            outletIndex[i, 0] = 1
            PoreBody[i, 6] += 1
            ThroatBody = np.r_[ThroatBody,
                               [[-1, i, 0, bin_avg_volume[bin_idx], bin_avg_radius[bin_idx], bin_avg_shape[bin_idx]]]]
            if r > 1E-4:
                print(i, bin_avg_volume[bin_idx], bin_avg_radius[bin_idx], bin_avg_shape[bin_idx])

    # --------------------------- Generate 4 output .dat file for pore and throat parameters ---------------------------
    # The structure of each .dat file is according to "pnextract_network_format" Maximal ball Ali Raeini
    Pore = np.copy(PoreBody)
    Link = np.copy(ThroatBody)
    Inlet = np.copy(inletIndex)
    Outlet = np.copy(outletIndex)
    Link = np.insert(Link, 0, list(range(0, len(Link))), axis=1)  # Add index of the throats
    folder = r'C:\4_fullscale\efficiency_new\PN2000'
    prefix = 'CB_2000_full_' + str(T + 1)

    T_node1_start = time.time()
    # Creating node1.dat file
    node1 = "{}\{}_node1.dat".format(folder, prefix)
    fN1 = open(node1, "w+")
    fN1.write('{}     {:.6E}     {:.6E}     {:.6E}\n'.format(total_pore_num, Lx, Ly, Lz))
    Link[:, 0:3] += 1
    Pore[:, 0] += 1

    p1s = Link[:, 1].astype(int)
    p2s = Link[:, 2].astype(int)
    adj = [[] for _ in range(total_pore_num + 1)]
    for thr_idx, (p1, p2) in enumerate(zip(p1s, p2s)):
        adj[p1].append(thr_idx)
        adj[p2].append(thr_idx)
    for i in range(1, total_pore_num + 1):
        thr_list = adj[i]
        base = len(thr_list)
        export = np.zeros([1, 2 * base + 2])
        for ii, thr_idx in enumerate(thr_list):
            p_row = Link[thr_idx, 1:3].astype(int)
            export[0, ii] = p_row[0] if p_row[1] == i else p_row[1]

        if int(Inlet[i - 1]) == 1:
            export[0, base] = 1
            export[0, base + 1] = 0
        elif int(Outlet[i - 1]) == 1:
            export[0, base] = 0
            export[0, base + 1] = 1
        else:
            export[0, base] = 0
            export[0, base + 1] = 0
        export[0, base + 2:] = np.array(thr_list, dtype=int) + 1
        export = np.asarray(export, dtype=int)
        str_export = ''

        for x in range(len(export[0, :])):
            str_export += str(export[0, x]) + '   '
        fN1.write('{}  {:.6E}  {:.6E}  {:.6E}  {}   {}\n'.format(int(Pore[i - 1, 0]), Pore[i - 1, 1], Pore[i - 1, 2],
                                                                 Pore[i - 1, 3], int(Pore[i - 1, 6]), str_export))

    fN1.close()
    T_node1_end = time.time() - T_node1_start
    print('Run time for writing node1 is: ', T_node1_end)

    T_rest_start = time.time()
    # Creating node2.dat file
    node2 = "{}\{}_node2.dat".format(folder, prefix)
    fN2 = open(node2, "w+")
    for i in range(len(Pore)):
        fN2.write('{}  {:.6E}  {:.6E}  {:.6E}  0\n'.format(int(Pore[i, 0]), Pore[i, 4], Pore[i, 5], Pore[i, 7]))
    fN2.close()

    # Creating link1.dat file
    link1 = "{}\{}_link1.dat".format(folder, prefix)
    fL1 = open(link1, "w+")
    fL1.write('{}\n'.format(len(Link)))
    for j in range(len(Link)):
        if Link[j, 1] > 0:
            p1 = Pore[int(Link[j, 1]) - 1, 1:4]  # Coordination of pore 1
            p2 = Pore[int(Link[j, 2]) - 1, 1:4]  # Coordination of pore 2
            d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (
                    p1[2] - p2[2]) ** 2)  # Throat total length(pore center to pore center)
            fL1.write('{}  {}  {}  {:.6E}  {:.6E}  {:.6E}\n'.format(int(Link[j, 0]), int(Link[j, 1]), int(Link[j, 2]),
                                                                    Link[j, 5], Link[j, 6], d))
        elif int(Link[j, 1]) == -1:
            d = Pore[int(Link[j, 2]) - 1, 1]
            fL1.write('{}  {}  {}  {:.6E}  {:.6E}  {:.6E}\n'.format(int(Link[j, 0]), int(Link[j, 1]), int(Link[j, 2]),
                                                                    Link[j, 5], Link[j, 6], d))
        elif int(Link[j, 1]) == 0:
            d = Lx - Pore[int(Link[j, 2]) - 1, 1]
            fL1.write('{}  {}  {}  {:.6E}  {:.6E}  {:.6E}\n'.format(int(Link[j, 0]), int(Link[j, 1]), int(Link[j, 2]),
                                                                    Link[j, 5], Link[j, 6], d))
    fL1.close()

    # Creating link2.dat file
    link2 = "{}\{}_link2.dat".format(folder, prefix)
    fL2 = open(link2, "w+")
    for k in range(len(Link)):
        if Link[k, 1] > 0:
            p1 = Pore[int(Link[k, 1]) - 1, 1:4]
            p2 = Pore[int(Link[k, 2]) - 1, 1:4]
            d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
            LP1 = (d / 2) * (1 - 0.6 * Link[k, 5] / Pore[int(Link[k, 1]) - 1, 5])  # Length of pore 1
            LP2 = (d / 2) * (1 - 0.6 * Link[k, 5] / Pore[int(Link[k, 2]) - 1, 5])  # Length of pore 2
            LT = d - LP1 - LP2  # Length of throat
            fL2.write('{}  {}  {}  {:.6E}  {:.6E}  {:.6E}  {:.6E}  0\n'.format(int(Link[k, 0]), int(Link[k, 1]),
                                                                               int(Link[k, 2]), LP1, LP2, LT,
                                                                               Link[k, 4]))
        elif int(Link[k, 1]) == -1:
            d = Pore[int(Link[k, 2]) - 1, 1]
            LP1 = res
            LP2 = (d / 2) * (1 - 0.6 * Link[k, 3] / Pore[int(Link[k, 2]) - 1, 5])
            LT = np.abs(d - LP1 - LP2)
            fL2.write('{}  {}  {}  {:.6E}  {:.6E}  {:.6E}  {:.6E}  0\n'.format(int(Link[k, 0]), int(Link[k, 1]),
                                                                               int(Link[k, 2]), LP1, LP2, LT,
                                                                               Link[k, 4]))
        elif int(Link[k, 1]) == 0:
            d = Lx - Pore[int(Link[k, 2]) - 1, 1]
            LP1 = res
            LP2 = (d / 2) * (1 - 0.6 * Link[k, 3] / Pore[int(Link[k, 2]) - 1, 5])
            LT = np.abs(d - LP1 - LP2)
            fL2.write('{}  {}  {}  {:.6E}  {:.6E}  {:.6E}  {:.6E}  0\n'.format(int(Link[k, 0]), int(Link[k, 1]),
                                                                               int(Link[k, 2]), LP1, LP2, LT,
                                                                               Link[k, 4]))
    fL2.close()

    T_rest_end = time.time() - T_rest_start
    print('Run time for writing node1 is: ', T_rest_end)
    # -------------------------------------------------------------------------------------------------

T_out = time.time() - T_in
print('Total  run time for ', str(Test), 'test is:', T_out)
