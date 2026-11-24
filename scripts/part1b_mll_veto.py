"""Part 1b: the Part 1 analysis with the ATLAS J/psi veto (m_ll > 5 GeV for every SFOC pair).

Usage:  python scripts/part1b_mll_veto.py
Writes: results/part1b_mll_veto.txt, figures/m4l_with_mll_veto.png

Part 1 (the official Open Data notebook selection, 507 data events) is left untouched; this is a
SEPARATE, additional selection. Only the m_ll > 5 GeV veto is applied -- not the full ATLAS
selection (no lepton pT/isolation/impact-parameter requirements, no Delta R cuts, no m12/m34 windows).
The 5 GeV value is the one used in the published ATLAS H->ZZ*->4l analyses (arXiv:1712.02304,
arXiv:2306.11379); it was NOT tuned on this data. The MOTIVATION for applying it came from the
low-m(Z2) feature seen in z_peak_diagnostics.py, so it is not independent of the earlier look.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hzz4l.events import load_or_build_table  # noqa: E402

VETO = 5.0
EDGES = np.arange(80, 255, 5)
OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def pge(n, mu):
    return stats.poisson.sf(n - 1, mu)


def passes(df):
    """Veto passed: no same-flavour opposite-charge pair below VETO. Events with no SFOC pair pass."""
    return ~(df.min_mll < VETO)


def window_row(label, da, mc, lo, hi):
    d = da[(da.m4l >= lo) & (da.m4l < hi)]
    m = mc[(mc.m4l >= lo) & (mc.m4l < hi)]
    mu, err = m.weight.sum(), np.sqrt((m.weight ** 2).sum())
    sg = m[m.group == "Signal (mH=125)"].weight.sum()
    log(f"{label:18s} {lo:3d}-{hi:3d}: data={len(d):3d}  MC={mu:6.2f} +- {err:4.2f} (signal {sg:5.2f})  "
        f"data/MC={len(d) / mu:5.2f}  p(N>=obs)={pge(len(d), mu):.2g}")


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    df = load_or_build_table()
    da, mc = df[df.group == "data"], df[df.group != "data"]

    log(f"Part 1 selection:        data events = {len(da)}  (official notebook: 507)")
    da_v, mc_v = da[passes(da)], mc[passes(mc)]
    log(f"Part 1b (m_ll > {VETO:.0f} GeV): data events = {len(da_v)}  "
        f"(removed {len(da) - len(da_v)}); MC weighted yield {mc.weight.sum():.1f} -> {mc_v.weight.sum():.1f}")
    removed = mc[~passes(mc)]
    log(f"MC removed by the veto, by group: " + ", ".join(
        f"{g}: {s.weight.sum():.2f}" for g, s in removed.groupby("group")))

    log("\n--- Windows before / after the veto (MC statistical uncertainty only, Poisson for data) ---")
    for lo, hi in ((85, 95), (120, 130)):
        window_row("Part 1", da, mc, lo, hi)
        window_row("Part 1b (veto)", da_v, mc_v, lo, hi)

    # Does the veto fully explain the 85-95 excess? Test the remaining excess honestly.
    d = da_v[(da_v.m4l >= 85) & (da_v.m4l < 95)]
    m = mc_v[(mc_v.m4l >= 85) & (mc_v.m4l < 95)]
    log(f"\nRemaining 85-95 GeV excess after the veto: data {len(d)} vs MC {m.weight.sum():.2f} "
        f"(difference {len(d) - m.weight.sum():+.1f} events; ratio {len(d) / m.weight.sum():.2f})")

    # Per-bin table after the veto
    d_c, _ = np.histogram(da_v.m4l, bins=EDGES)
    m_c, _ = np.histogram(mc_v.m4l, bins=EDGES, weights=mc_v.weight)
    m_e = np.sqrt(np.histogram(mc_v.m4l, bins=EDGES, weights=mc_v.weight ** 2)[0])
    log("\nbin [GeV]   data   MC     MCerr   p(N>=obs)")
    for lo, a, b, e in zip(EDGES[:-1], d_c, m_c, m_e):
        if lo < 120 or lo in (120, 125):
            log(f"{lo:3d}-{lo + 5:3d}   {a:5d} {b:6.2f} {e:6.2f}   {pge(int(a), b) if b > 0 else float('nan'):.2g}")
    pmin = min(pge(int(a), b) for a, b in zip(d_c, m_c) if b > 0)
    log(f"smallest single-bin p over {len(d_c)} bins after the veto: {pmin:.2g} (x{len(d_c)} trials ~ {min(1, pmin * len(d_c)):.2g})")

    # Plot ------------------------------------------------------------------------------
    centres = EDGES[:-1] + 2.5
    groups = ["Z, ttbar bkg", "ZZ* bkg", "Signal (mH=125)"]
    colors = ["#6b59d3", "#ff0000", "#00cdff"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist([mc_v[mc_v.group == g].m4l for g in groups], bins=EDGES,
            weights=[mc_v[mc_v.group == g].weight for g in groups], stacked=True, color=colors, label=groups)
    ax.bar(centres, 2 * m_e, bottom=m_c - m_e, width=5, fill=False, hatch="////", edgecolor="gray", lw=0,
           label="MC stat. unc.")
    ax.errorbar(centres, d_c, yerr=np.sqrt(d_c), fmt="ko", label="Data")
    ax.set_xlim(80, 250)
    ax.set_xlabel("m4l [GeV]")
    ax.set_ylabel("Events / 5 GeV")
    ax.set_title(f"Part 1b: Part 1 selection + m_ll > {VETO:.0f} GeV veto (all SFOC pairs)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("figures/m4l_with_mll_veto.png", dpi=150)

    with open("results/part1b_mll_veto.txt", "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
