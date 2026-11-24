"""Data access for the ATLAS Open Data 4-lepton samples (2020 release, 13 TeV, 10 fb^-1).

All remote inputs are pinned: the cross-section table (infofile.py) is fetched from a fixed
commit of atlas-outreach-data-tools/notebooks-collection-opendata, not from `master`, so a
re-run cannot silently change because the upstream file changed.
"""
import importlib.util
import os
import time
import urllib.request

BASE = "https://opendata.cern.ch/eos/opendata/atlas/OutreachDatasets/2020-01-22/4lep/"
INFOFILE_COMMIT = "8509890af9a4db47f2c9908063c98d50b74123c2"   # last change to infofile.py (2021-02-24)
INFOFILE_URL = ("https://raw.githubusercontent.com/atlas-outreach-data-tools/"
                f"notebooks-collection-opendata/{INFOFILE_COMMIT}/"
                "13-TeV-examples/uproot_python/infofile.py")
LUMI = 10   # fb^-1 (data_A + data_B + data_C + data_D)

DATA_SAMPLES = ["data_A", "data_B", "data_C", "data_D"]
GROUPS = {
    "Z, ttbar bkg": ["Zee", "Zmumu", "ttbar_lep"],
    "ZZ* bkg": ["llll"],
    "Signal (mH=125)": ["ggH125_ZZ4lep", "VBFH125_ZZ4lep", "WH125_ZZ4lep", "ZH125_ZZ4lep"],
}

ID_BRANCHES = ["runNumber", "eventNumber", "channelNumber"]
BASE_BRANCHES = ["lep_pt", "lep_eta", "lep_phi", "lep_E", "lep_charge", "lep_type"]
MC_BRANCHES = ["mcWeight", "scaleFactor_PILEUP", "scaleFactor_ELE",
               "scaleFactor_MUON", "scaleFactor_LepTRIGGER"]


def _remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch(url, cache_dir="data_cache", tries=40):
    """Download a file once into cache_dir and return the local path.

    The download resumes from the bytes already in `<file>.part` (HTTP Range), so a dropped
    connection or a 503 does not restart a 180 MB file from zero. The finished file is only
    moved into place if its size matches the server's Content-Length.
    """
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    part = local + ".part"
    total = None
    for i in range(tries):
        try:
            if total is None:
                total = _remote_size(url)
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have > total:                      # corrupt leftover: start again
                os.remove(part)
                have = 0
            if have < total:
                req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    if have and r.status != 206:  # server ignored Range: restart cleanly
                        os.remove(part)
                        have = 0
                    with open(part, "ab" if have else "wb") as f:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
            if os.path.getsize(part) == total:
                os.replace(part, local)
                return local
            raise IOError(f"incomplete: {os.path.getsize(part)} of {total} bytes")
        except Exception as e:  # network errors: back off and resume
            print(f"    attempt {i + 1} failed: {e}")
            time.sleep(min(30, 3 * (i + 1)))
    raise RuntimeError(f"Could not download: {url}")


def load_infofile(cache_dir="data_cache"):
    """Import the pinned infofile module and return its `infos` dict."""
    path = fetch(INFOFILE_URL, cache_dir)
    spec = importlib.util.spec_from_file_location("infofile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.infos


def sample_url(name, is_data, infos):
    if is_data:
        return f"{BASE}Data/{name}.4lep.root"
    return f"{BASE}MC/mc_{infos[name]['DSID']}.{name}.4lep.root"
