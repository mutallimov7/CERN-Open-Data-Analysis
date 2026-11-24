"""Per-event table for the selected 4-lepton events (data and simulation).

Columns: sample, group, period (data only), channel, m4l, Z1/Z2 masses, sorted lepton pT, weight.
"""
import os

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .data import (BASE_BRANCHES, DATA_SAMPLES, GROUPS, ID_BRANCHES, MC_BRANCHES, fetch,
                   load_infofile, sample_url)
from .selection import event_weight, four_vectors, selection_mask

M_Z = 91.1876
CHANNELS = {44: "4e", 48: "2e2mu", 52: "4mu"}


def _inv_mass(a, b):
    """Invariant mass of the sum of two (E, px, py, pz) arrays of shape (N, 4)."""
    s = a + b
    m2 = s[:, 0] ** 2 - s[:, 1] ** 2 - s[:, 2] ** 2 - s[:, 3] ** 2
    return np.sqrt(np.clip(m2, 0, None))


def z_pair_masses(p4, charge, lep_type):
    """Z1 / Z2 masses from the same-flavour opposite-charge pairing with a pair closest to m_Z.

    p4: (N, 4, 4) array of (E, px, py, pz) in GeV; charge, lep_type: (N, 4).
    Returns (m_z1, m_z2); NaN if no valid pairing exists. This DOES happen after the selection
    (total charge 0 and type sum 48 also admit e+e+mu-mu-, which has no same-flavour
    opposite-charge pair), so callers must not drop NaN rows silently.
    """
    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    n = len(p4)
    best_dist = np.full(n, np.inf)
    m1 = np.full(n, np.nan)
    m2 = np.full(n, np.nan)

    def valid(i, j):
        return (charge[:, i] + charge[:, j] == 0) & (lep_type[:, i] == lep_type[:, j])

    for (a, b), (c, d) in pairings:
        ok = valid(a, b) & valid(c, d)
        mab = _inv_mass(p4[:, a], p4[:, b])
        mcd = _inv_mass(p4[:, c], p4[:, d])
        for ma, mb in ((mab, mcd), (mcd, mab)):          # either pair may be Z1
            dist = np.abs(ma - M_Z)
            take = ok & (dist < best_dist)
            best_dist = np.where(take, dist, best_dist)
            m1 = np.where(take, ma, m1)
            m2 = np.where(take, mb, m2)
    return m1, m2


def min_sfos_mass(p4, charge, lep_type):
    """Smallest invariant mass over ALL same-flavour opposite-charge lepton pairs (6 pairs of 4).

    This is the quantity the ATLAS J/psi veto acts on (m_ll > 5 GeV for every SFOC pair).
    NaN if the event has no such pair.
    """
    out = np.full(len(p4), np.inf)
    for i in range(4):
        for j in range(i + 1, 4):
            ok = (charge[:, i] + charge[:, j] == 0) & (lep_type[:, i] == lep_type[:, j])
            m = _inv_mass(p4[:, i], p4[:, j])
            out = np.where(ok & (m < out), m, out)
    return np.where(np.isinf(out), np.nan, out)


def load_sample(name, is_data, infos, cache_dir="data_cache"):
    branches = ID_BRANCHES + BASE_BRANCHES + ([] if is_data else MC_BRANCHES)
    path = fetch(sample_url(name, is_data, infos), cache_dir)
    with uproot.open(path + ":mini") as tree:
        ev = tree.arrays(branches)
    n_in = len(ev)
    mask, type_sum = selection_mask(ev)
    ev, type_sum = ev[mask], ak.to_numpy(type_sum[mask])

    p4 = four_vectors(ev)
    total = ak.sum(p4, axis=1)
    pt = ak.to_numpy(ak.to_regular(p4.pt, axis=1))
    order = np.argsort(-pt, axis=1)
    pt_sorted = np.take_along_axis(pt, order, axis=1)

    arr = np.stack([ak.to_numpy(ak.to_regular(x, axis=1)) for x in (p4.E, p4.px, p4.py, p4.pz)], axis=-1)
    charge = ak.to_numpy(ak.to_regular(ev.lep_charge[:, :4], axis=1))
    ltype = ak.to_numpy(ak.to_regular(ev.lep_type[:, :4], axis=1))
    mz1, mz2 = z_pair_masses(arr, charge, ltype)
    min_mll = min_sfos_mass(arr, charge, ltype)

    info = None if is_data else infos[name]
    df = pd.DataFrame({
        "sample": name,
        "period": name if is_data else "",
        "channel": [CHANNELS[t] for t in type_sum],
        "m4l": ak.to_numpy(total.mass),
        "mz1": mz1, "mz2": mz2, "min_mll": min_mll,
        "pt1": pt_sorted[:, 0], "pt2": pt_sorted[:, 1], "pt3": pt_sorted[:, 2], "pt4": pt_sorted[:, 3],
        "runNumber": ak.to_numpy(ev.runNumber),
        "weight": event_weight(info, ev, is_data),
    })
    print(f"  {name}: nIn={n_in}, nOut={len(df)}")
    return df


def load_or_build_table(path="data_cache/event_table.pkl", cache_dir="data_cache"):
    """Return the cached event table, rebuilding it if it is missing or has an older column set."""
    required = {"min_mll", "mz1", "mz2", "m4l", "weight", "group", "channel", "period"}
    if os.path.exists(path):
        df = pd.read_pickle(path)
        if required <= set(df.columns):
            return df
        print("cached event table is out of date; rebuilding")
    df = build_event_table(cache_dir)
    df.to_pickle(path)
    return df


def build_event_table(cache_dir="data_cache"):
    infos = load_infofile(cache_dir)
    parts = []
    for name in DATA_SAMPLES:
        d = load_sample(name, True, infos, cache_dir)
        d["group"] = "data"
        parts.append(d)
    for group, names in GROUPS.items():
        for name in names:
            d = load_sample(name, False, infos, cache_dir)
            d["group"] = group
            parts.append(d)
    return pd.concat(parts, ignore_index=True)
