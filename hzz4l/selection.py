"""Event selection, four-vectors and weights (identical logic to the notebook / official ATLAS notebook)."""
import awkward as ak
import numpy as np

from .data import LUMI


def selection_mask(ev):
    """First four leptons, total charge 0, flavour types eeee / mumumumu / eemumu.

    Returns (mask, type_sum); type_sum is 44 = 4e, 48 = 2e2mu, 52 = 4mu.
    """
    charge_sum = ak.sum(ev.lep_charge[:, :4], axis=1)
    type_sum = ak.sum(ev.lep_type[:, :4], axis=1)
    mask = (charge_sum == 0) & ((type_sum == 44) | (type_sum == 48) | (type_sum == 52))
    return mask, type_sum


def four_vectors(ev):
    """Four-vectors of the first four leptons, in GeV (input is in MeV)."""
    import vector
    vector.register_awkward()
    return ak.zip({"pt": ev.lep_pt[:, :4] / 1000, "eta": ev.lep_eta[:, :4],
                   "phi": ev.lep_phi[:, :4], "E": ev.lep_E[:, :4] / 1000},
                  with_name="Momentum4D")


def event_weight(info, ev, is_data):
    """1 for data; cross-section x luminosity x scale factors for simulation."""
    if is_data:
        return np.ones(len(ev))
    xsec_weight = LUMI * 1000 * info["xsec"] / (info["sumw"] * info["red_eff"])
    return ak.to_numpy(xsec_weight * ev.mcWeight * ev.scaleFactor_PILEUP *
                       ev.scaleFactor_ELE * ev.scaleFactor_MUON * ev.scaleFactor_LepTRIGGER)
