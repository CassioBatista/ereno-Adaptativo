"""Run ReSIDS v2 (~/ereno-Adaptativo, ereno.py distributed) unchanged, with the lean loader
of v2_loader.py (columns of the run's features file), on any evaluation-v1 domain.
  python run_v2.py <conf.yaml> <num_clients>
Part of the repository since 2026-11-25 (scripts/ev1_tl_runners)."""
import json
import os
import runpy
import sys

import yaml

here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.expanduser("~/ereno-Adaptativo")
conf_path, n = os.path.abspath(sys.argv[1]), sys.argv[2]
conf = yaml.safe_load(open(conf_path))
os.chdir(repo)
sys.path[:0] = [repo, os.path.join(repo, "scripts"), here]
import python.util as util  # noqa: E402
from v2_loader import make_loader  # noqa: E402
feats = json.load(open(conf["grasp"]["features_file"]))["features"]
util.load_arff = make_loader(feats)
sys.argv = ["ereno.py", "distributed", "GR-G-VND", "6", conf["dataset"]["name"],
            "--conf", conf_path, "--clients", n]
runpy.run_path(os.path.join(repo, "ereno.py"), run_name="__main__")
