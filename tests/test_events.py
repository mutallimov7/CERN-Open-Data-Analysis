import numpy as np
import pytest

from hzz4l.events import M_Z, z_pair_masses


def _back_to_back(m, axis=0):
    """Two massless leptons with invariant mass m, back to back along `axis` (1=x, 2=y, 3=z)."""
    a = np.zeros(4)
    b = np.zeros(4)
    a[0] = b[0] = m / 2
    a[axis + 1] = m / 2
    b[axis + 1] = -m / 2
    return a, b


def test_pair_masses_picks_pair_closest_to_mz():
    # 2e2mu event: e+e- pair with m = 90, mu+mu- pair with m = 20
    e1, e2 = _back_to_back(90.0, axis=0)
    m1, m2 = _back_to_back(20.0, axis=1)
    p4 = np.stack([e1, e2, m1, m2])[None]                 # (1, 4, 4)
    charge = np.array([[1, -1, 1, -1]])
    ltype = np.array([[11, 11, 13, 13]])
    z1, z2 = z_pair_masses(p4, charge, ltype)
    assert z1[0] == pytest.approx(90.0)
    assert z2[0] == pytest.approx(20.0)


def test_pair_masses_swaps_when_second_pair_is_closer():
    a1, a2 = _back_to_back(15.0, axis=0)
    b1, b2 = _back_to_back(88.0, axis=1)
    p4 = np.stack([a1, a2, b1, b2])[None]
    charge = np.array([[1, -1, 1, -1]])
    ltype = np.array([[13, 13, 11, 11]])
    z1, z2 = z_pair_masses(p4, charge, ltype)
    assert z1[0] == pytest.approx(88.0)
    assert z2[0] == pytest.approx(15.0)


def test_min_sfos_mass_uses_all_pairs_not_only_the_chosen_pairing():
    # 4mu event; leptons 0,1 form a 90 GeV pair and 2,3 a 40 GeV pair, but the CROSS pairing
    # (0,3) has opposite charge and a tiny mass (collinear): the veto must see the 2 GeV pair.
    from hzz4l.events import min_sfos_mass
    p = np.array([
        [45.0, 45.0, 0.0, 0.0],     # lepton 0 (E, px, py, pz)
        [45.0, -45.0, 0.0, 0.0],    # lepton 1
        [20.0, 0.0, 20.0, 0.0],     # lepton 2
        [20.0, 0.0, -20.0, 0.0],    # lepton 3
    ])
    # make lepton 3 almost collinear with lepton 0 so that m(0,3) is small
    p[3] = [45.0, 44.99, 0.5, 0.0]
    p4 = p[None]
    charge = np.array([[1, -1, 1, -1]])
    ltype = np.array([[13, 13, 13, 13]])
    m = min_sfos_mass(p4, charge, ltype)[0]
    assert m < 5.0


def test_min_sfos_mass_nan_without_sfos_pair():
    from hzz4l.events import min_sfos_mass
    p4 = np.ones((1, 4, 4))
    charge = np.array([[1, 1, -1, -1]])
    ltype = np.array([[11, 11, 13, 13]])     # e+ e+ mu- mu-: no same-flavour opposite-charge pair
    assert np.isnan(min_sfos_mass(p4, charge, ltype)[0])


def test_mz_constant():
    assert M_Z == pytest.approx(91.1876)
