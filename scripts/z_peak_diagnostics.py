"""Diagnostics for the data excess at 85-95 GeV in m4l (the README's 'open question').

Usage:  python scripts/z_peak_diagnostics.py
Writes: results/z_peak_diagnostics.txt and figures/zpeak_*.png

Everything below uses MC statistical uncertainty only (no systematics) and Poisson statistics
for the data. The window and the tests are fixed BEFORE looking at the sub-splits; the
sub-splits (channel, period) are descriptive, not independent discoveries.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hzz4l.events import load_or_build_table  # noqa: E402

TABLE = "data_cache/event_table.pkl"
WIN = (85.0, 95.0)
OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def poisson_p_ge(n_obs, mu):
    """P(N >= n_obs) for N ~ Poisson(mu)."""
    return stats.poisson.sf(n_obs - 1, mu)


def garwood(n, cl=0.6827):
    a = 1 - cl
    lo = 0.0 if n == 0 else stats.chi2.ppf(a / 2, 2 * n) / 2
    hi = stats.chi2.ppf(1 - a / 2, 2 * (n + 1)) / 2
    return lo, hi


def window(df, lo=WIN[0], hi=WIN[1]):
    return df[(df.m4l >= lo) & (df.m4l < hi)]


def summarise(df, by, title):
    log(f"\n--- {title} ({WIN[0]:.0f}-{WIN[1]:.0f} GeV) ---")
    log(f"{by:12s} {'data':>6s} {'MC':>8s} {'MCerr':>7s} {'data/MC':>8s} {'68% CI on ratio':>18s} {'p(N>=obs)':>10s}")
    w = window(df)
    for key in sorted(w[by].unique()):
        d = w[(w[by] == key) & (w.group == "data")]
        m = w[(w[by] == key) & (w.group != "data")]
        n, mu = len(d), m.weight.sum()
        err = np.sqrt((m.weight ** 2).sum())
        if mu > 0:
            lo, hi = garwood(n)
            log(f"{key:12s} {n:6d} {mu:8.2f} {err:7.2f} {n / mu:8.2f} {f'[{lo / mu:.2f}, {hi / mu:.2f}]':>18s} {poisson_p_ge(n, mu):10.2g}")


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    df = load_or_build_table(TABLE)

    mc = df[df.group != "data"]
    da = df[df.group == "data"]
    log(f"Selected data events: {len(da)} (official notebook: 507)")

    # 1. Overall window ---------------------------------------------------------------
    w_d, w_m = window(da), window(mc)
    n, mu = len(w_d), w_m.weight.sum()
    err = np.sqrt((w_m.weight ** 2).sum())
    log(f"\nWindow {WIN}: data={n}, MC={mu:.2f} +- {err:.2f} (stat), data/MC={n / mu:.2f}")
    log(f"Poisson p(N>={n} | mu={mu:.2f}) = {poisson_p_ge(n, mu):.2g}  (MC stat only, no systematics)")
    # Look-elsewhere: the window is centred on m_Z by hypothesis, but report the scan too
    edges = np.arange(80, 255, 5)
    d_c, _ = np.histogram(da.m4l, bins=edges)
    m_c, _ = np.histogram(mc.m4l, bins=edges, weights=mc.weight)
    pmin = min(poisson_p_ge(int(a), b) for a, b in zip(d_c, m_c) if b > 0)
    nb = len(d_c)
    log(f"Smallest single-bin Poisson p over {nb} bins of 5 GeV: {pmin:.2g} "
        f"(x{nb} trials ~ {min(1, pmin * nb):.2g})")

    # 2. Where does the MC in the window come from? -----------------------------------
    log(f"\n--- MC yield in the window by group ---")
    for g, s in w_m.groupby("group"):
        log(f"{g:18s} {s.weight.sum():8.2f} +- {np.sqrt((s.weight ** 2).sum()):.2f}  ({len(s)} raw events)")
    # 3. By channel, by period ---------------------------------------------------------
    summarise(df, "channel", "by channel")
    log("\n--- data by period (events in window / all selected events) ---")
    for p in sorted(da.period.unique()):
        dp = da[da.period == p]
        log(f"{p}: {len(window(dp))} / {len(dp)}  ({len(window(dp)) / len(dp):.3f})")
    log(f"(weighted MC fraction in window: {w_m.weight.sum() / mc.weight.sum():.3f}; "
        f"data overall: {len(w_d) / len(da):.3f})")

    # 4. Pair masses: Z1 ~ m_Z and soft Z2 would indicate Z -> 4l (Z -> ll + gamma* -> ll) ----
    log("\n--- Z1/Z2 pair masses and softest-lepton pT in the window (medians) ---")
    for name, d in (("data", w_d), ("MC (weighted)", w_m)):
        if name == "data":
            med = lambda c: np.median(d[c])
        else:
            def med(c, d=d):
                o = np.argsort(d[c].to_numpy())
                cw = np.cumsum(d.weight.to_numpy()[o])
                return d[c].to_numpy()[o][np.searchsorted(cw, cw[-1] / 2)]
        log(f"{name:14s} mZ1={med('mz1'):.1f}  mZ2={med('mz2'):.1f}  pt4={med('pt4'):.1f}  pt3={med('pt3'):.1f}")

    # 4b. Low m(Z2) feature (added after the medians above looked identical) ----------------
    # NOTE: these cuts were chosen AFTER seeing the m(Z2) plot, so p-values are descriptive.
    log("\n--- Low m(Z2): data vs MC in the window, and MC edge (all m4l) ---")
    for lo, hi in [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 12), (12, 20), (20, 60)]:
        d = w_d[(w_d.mz2 >= lo) & (w_d.mz2 < hi)]
        m = w_m[(w_m.mz2 >= lo) & (w_m.mz2 < hi)]
        log(f"mZ2 {lo:2d}-{hi:2d}: data={len(d):3d}  MC={m.weight.sum():6.2f} +- {np.sqrt((m.weight ** 2).sum()):.2f} (raw {len(m)})")
    bins1 = np.arange(0, 12, 1)
    zz = mc[mc.group == "ZZ* bkg"]
    log("ZZ* MC, all m4l, mZ2 per 1 GeV (0..11):", np.round(np.histogram(zz.mz2, bins1, weights=zz.weight)[0], 2).tolist())
    log("data,   all m4l, mZ2 per 1 GeV (0..11):", np.histogram(da.mz2, bins1)[0].tolist())
    fine = np.arange(2.5, 4.01, 0.25)
    log("data, all m4l, mZ2 per 0.25 GeV (2.5..4.0):", np.histogram(da.mz2, fine)[0].tolist(),
        "(J/psi mass = 3.097 GeV)")
    log("data, all m4l, mZ2 in [2.9, 3.3): %d;  [2.5, 2.9): %d;  [3.3, 3.7): %d" % (
        ((da.mz2 >= 2.9) & (da.mz2 < 3.3)).sum(), ((da.mz2 >= 2.5) & (da.mz2 < 2.9)).sum(),
        ((da.mz2 >= 3.3) & (da.mz2 < 3.7)).sum()))
    log("\n--- Window 85-95 and Higgs window 120-130 with and without low m(Z2) ---")
    for name, lo_, hi_ in (("85-95", 85, 95), ("120-130", 120, 130)):
        for label, cut in (("all", "all"), ("no SFOS", "nan"), ("mZ2<5", None), ("mZ2>=5", 5), ("mZ2>=12", 12)):
            if cut == "all":
                sel = lambda d: (d.m4l >= lo_) & (d.m4l < hi_)
            elif cut == "nan":
                sel = lambda d: (d.m4l >= lo_) & (d.m4l < hi_) & d.mz2.isna()
            elif cut is None:
                sel = lambda d: (d.m4l >= lo_) & (d.m4l < hi_) & (d.mz2 < 5)
            else:
                sel = lambda d, c=cut: (d.m4l >= lo_) & (d.m4l < hi_) & (d.mz2 >= c)
            d, m = da[sel(da)], mc[sel(mc)]
            mu = m.weight.sum()
            sg = m[m.group == "Signal (mH=125)"].weight.sum()
            log(f"{name:8s} {label:8s}: data={len(d):3d}  MC total={mu:6.2f} (of which signal {sg:5.2f})  "
                f"data/MC={len(d) / mu:5.2f}  p(N>=obs)={poisson_p_ge(len(d), mu):.2g}")

    # 5. Plots -------------------------------------------------------------------------
    def compare(col, bins, xlabel, fname, sel=lambda x: x):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        wm, wd = sel(w_m), sel(w_d)
        ax.hist(wm[col], bins=bins, weights=wm.weight, histtype="stepfilled", color="#ff9999",
                edgecolor="red", label="MC (all, weighted)")
        h, e = np.histogram(wd[col], bins=bins)
        c = 0.5 * (e[1:] + e[:-1])
        ax.errorbar(c, h, yerr=np.sqrt(h), fmt="ko", label="Data")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Events")
        ax.set_title(f"m4l in {WIN[0]:.0f}-{WIN[1]:.0f} GeV")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"figures/{fname}", dpi=150)
        plt.close(fig)

    compare("mz2", np.arange(0, 60, 4), "m(Z2) [GeV]", "zpeak_mz2.png")
    compare("pt4", np.arange(5, 45, 3), "softest lepton pT [GeV]", "zpeak_pt4.png")
    compare("mz1", np.arange(60, 120, 4), "m(Z1) [GeV]", "zpeak_mz1.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, ch in zip(axes, ["4e", "2e2mu", "4mu"]):
        wm, wd = w_m[w_m.channel == ch], w_d[w_d.channel == ch]
        bins = np.arange(*WIN, 2.5)
        ax.hist(wm.m4l, bins=bins, weights=wm.weight, histtype="stepfilled", color="#ff9999", edgecolor="red")
        h, e = np.histogram(wd.m4l, bins=bins)
        ax.errorbar(0.5 * (e[1:] + e[:-1]), h, yerr=np.sqrt(h), fmt="ko")
        ax.set_title(f"{ch}: data {len(wd)} / MC {wm.weight.sum():.1f}")
        ax.set_xlabel("m4l [GeV]")
    fig.tight_layout()
    fig.savefig("figures/zpeak_channels.png", dpi=150)
    plt.close(fig)

    with open("results/z_peak_diagnostics.txt", "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
