"""Signal-strength fit for H -> ZZ* -> 4l with pyhf (profile likelihood).

Usage:  python scripts/pyhf_fit.py [part1|part1b]     (default: part1b)
          part1  : the official Open Data selection (507 data events)
          part1b : Part 1 + ATLAS J/psi veto m_ll > 5 GeV for all SFOC pairs (see part1b_mll_veto.py)
Writes: results/pyhf_fit_<sel>.txt, figures/pyhf_profile_<sel>.png

One channel, m4l bins of 5 GeV in [105, 160]. Model:
  signal      = ggH + VBF + WH + ZH, scaled by the signal strength mu (normfactor)
  background  = ZZ* (llll) + Z/ttbar, with MC statistical uncertainty (staterror)
Illustrative nuisance parameters -- these values are ASSUMPTIONS for demonstration, not
official ATLAS uncertainties:
  lumi   : +-2% on signal and all background (normsys)
  ZZ*    : +-10% normalisation on the llll sample (normsys)
Educational dataset: the numbers are not a physics result.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hzz4l.events import load_or_build_table  # noqa: E402

EDGES = np.arange(105, 165, 5)
LUMI_UNC, ZZ_UNC = 0.02, 0.10
OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def hist(df):
    h, _ = np.histogram(df.m4l, bins=EDGES, weights=df.weight)
    h2, _ = np.histogram(df.m4l, bins=EDGES, weights=df.weight ** 2)
    return h, np.sqrt(h2)


def build_workspace(df):
    sig, sig_err = hist(df[df.group == "Signal (mH=125)"])
    zz, zz_err = hist(df[df.group == "ZZ* bkg"])
    zt, zt_err = hist(df[df.group == "Z, ttbar bkg"])
    data, _ = np.histogram(df[df.group == "data"].m4l, bins=EDGES)

    # Z/ttbar has very few raw events: floor the MC uncertainty so empty bins are not 'perfect'
    zt_err = np.maximum(zt_err, 1e-3)
    spec = {
        "channels": [{
            "name": "SR",
            "samples": [
                {"name": "signal", "data": sig.tolist(), "modifiers": [
                    {"name": "mu", "type": "normfactor", "data": None},
                    {"name": "lumi", "type": "normsys", "data": {"hi": 1 + LUMI_UNC, "lo": 1 - LUMI_UNC}},
                    {"name": "staterror_sig", "type": "staterror", "data": sig_err.tolist()}]},
                {"name": "ZZ", "data": zz.tolist(), "modifiers": [
                    {"name": "lumi", "type": "normsys", "data": {"hi": 1 + LUMI_UNC, "lo": 1 - LUMI_UNC}},
                    {"name": "zz_norm", "type": "normsys", "data": {"hi": 1 + ZZ_UNC, "lo": 1 - ZZ_UNC}},
                    {"name": "staterror_zz", "type": "staterror", "data": zz_err.tolist()}]},
                {"name": "Zttbar", "data": zt.tolist(), "modifiers": [
                    {"name": "lumi", "type": "normsys", "data": {"hi": 1 + LUMI_UNC, "lo": 1 - LUMI_UNC}},
                    {"name": "staterror_zt", "type": "staterror", "data": zt_err.tolist()}]},
            ]}],
        "observations": [{"name": "SR", "data": data.tolist()}],
        "measurements": [{"name": "fit", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }
    return pyhf.Workspace(spec), data, sig, zz, zt


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else "part1b"
    if sel not in ("part1", "part1b"):
        sys.exit("selection must be part1 or part1b")
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    df = load_or_build_table()
    if sel == "part1b":
        df = df[~(df.min_mll < 5.0)]          # events with no SFOC pair pass, as in part1b_mll_veto.py
    log(f"Selection: {sel}; data events in table: {(df.group == 'data').sum()}")
    ws, data, sig, zz, zt = build_workspace(df)
    model = ws.model()
    obs = ws.data(model)

    log(f"pyhf {pyhf.__version__}; bins {EDGES[0]}-{EDGES[-1]} GeV in 5 GeV steps")
    log("bin_low  data   signal    ZZ*     Z/ttbar")
    for e, d, s, z, t in zip(EDGES[:-1], data, sig, zz, zt):
        log(f"{e:7d} {d:5d} {s:8.2f} {z:8.2f} {t:8.2f}")
    log(f"total   {data.sum():5d} {sig.sum():8.2f} {zz.sum():8.2f} {zt.sum():8.2f}")

    # Best fit and uncertainty ---------------------------------------------------------
    bestfit, twice_nll = pyhf.infer.mle.fit(obs, model, return_fitted_val=True)
    mu_hat = bestfit[model.config.poi_index]
    grid = np.linspace(0, 4, 81)
    nll = []
    for m in grid:
        _, val = pyhf.infer.mle.fixed_poi_fit(m, obs, model, return_fitted_val=True)
        nll.append(float(val))
    nll = np.array(nll)
    dnll = nll - float(twice_nll)
    inside = grid[dnll <= 1.0]                   # 68% CL: delta(-2 ln L) <= 1
    lo, hi = inside.min(), inside.max()
    log(f"\nBest-fit signal strength mu = {mu_hat:.2f}  (68% interval from profile: [{lo:.2f}, {hi:.2f}]; "
        f"grid step 0.05, [0,4])")

    # Discovery significance for mu = 0 ---------------------------------------------------
    p0 = pyhf.infer.hypotest(0.0, obs, model, test_stat="q0")
    from scipy import stats
    z0 = stats.norm.isf(float(p0))
    log(f"Observed significance of the background-only rejection: p0 = {float(p0):.3g}, Z = {z0:.2f} sigma")
    # Expected significance for mu = 1 (Asimov dataset, nuisance parameters at their nominal values)
    init, bounds, fixed = model.config.suggested_init(), model.config.suggested_bounds(), model.config.suggested_fixed()
    asimov_pars = list(init)
    asimov_pars[model.config.poi_index] = 1.0
    asimov = model.expected_data(asimov_pars)
    q0_a = float(pyhf.infer.test_statistics.q0(0.0, asimov, model, init, bounds, fixed))
    log(f"Expected significance for mu = 1 (Asimov): Z = {np.sqrt(max(q0_a, 0.0)):.2f} sigma")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, dnll)
    ax.axhline(1, ls="--", c="gray")
    ax.set_ylim(0, 8)
    ax.set_xlabel("signal strength mu")
    ax.set_ylabel(r"$-2\Delta\ln L$")
    ax.set_title("Profile likelihood (illustrative nuisance parameters)")
    fig.tight_layout()
    fig.savefig(f"figures/pyhf_profile_{sel}.png", dpi=150)

    with open(f"results/pyhf_fit_{sel}.txt", "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
