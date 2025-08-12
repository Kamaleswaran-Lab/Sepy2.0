import os
import sys
import glob
import pickle
import random
from pathlib import Path

import pandas as pd

# Ensure local modules are importable
sys.path.append("/hpc/home/ma618/Sepy/")

import utils 
import sepyDICT as sd  


# =====================
# User-configurable year
# =====================
YEAR = 2019  # <-- set the year you want to test

# Optional: set custom config paths if different from defaults
DATA_CONFIG_PATH = "/hpc/home/ma618/Sepy/configurations/emory_config.yaml"
SEPY_CONFIG_PATH = "/hpc/home/ma618/Sepy/configurations/dict_config.yaml"


def _get_variable_chart_path(data_config: dict) -> str:
    """
    Resolve the path to the variable chart CSV using the grouping config.
    Mirrors the logic used in make_dicts.py for grouping_types.
    """
    groupings_root = Path(os.path.expandvars(data_config["groupings_path"]))
    grouping_types = data_config["dictionary_paths"]["grouping_types"]

    variable_chart_fp = None
    for grouping_type, grouping_glob in grouping_types:
        if grouping_type == "variable_chart":
            matches = glob.glob(str(groupings_root / grouping_glob))
            if matches:
                variable_chart_fp = matches[0]
            break

    if not variable_chart_fp:
        raise FileNotFoundError(
            "Could not resolve 'variable_chart' path from grouping_types."
        )
    return variable_chart_fp


def _get_yearly_pickle_path(data_config: dict, year: int) -> str:
    yearly_dir = Path(os.path.expandvars(data_config["yearly_pickle_output_path"]))
    yearly_fp = yearly_dir / f"{data_config['dataset_identifier']}{year}.pickle"
    return str(yearly_fp)


def main() -> None:
    # Load configs
    data_config = utils.load_yaml(DATA_CONFIG_PATH)
    sepy_config = utils.load_yaml(SEPY_CONFIG_PATH)

    # Resolve and load yearly pickle
    yearly_pickle_fp = _get_yearly_pickle_path(data_config, YEAR)
    if not os.path.exists(yearly_pickle_fp):
        raise FileNotFoundError(f"Yearly pickle not found: {yearly_pickle_fp}")
    
    print(f"Yearly pickle path: {yearly_pickle_fp}")
    with open(yearly_pickle_fp, "rb") as f:
        import_instance = pickle.load(f)
    print("Read the yearly pickle")

    # Resolve bounds (variable_chart)
    variable_chart_fp = _get_variable_chart_path(data_config)
    bounds = pd.read_csv(variable_chart_fp)

    # Create a save_dir placeholder (will not be used for saving in this test)
    supertable_root = Path(os.path.expandvars(data_config["supertable_output_path"]))
    save_dir = supertable_root / str(YEAR)

    # Create sepyMaster
    sepy_master = sd.sepyMaster(import_instance, sepy_config, bounds, str(save_dir))

    # Pick a random CSN from the imported encounters
    if not getattr(import_instance, "csns", None):
        raise ValueError("No CSNs found in the yearly import instance.")

    csn = random.choice(import_instance.csns)
    print(f"Selected CSN: {csn}")
    import pdb; pdb.set_trace()

    # Create a CSN instance and generate the supertable WITHOUT saving
    csn_instance = sepy_master.create_csn_instance(csn)
    csn_instance.create_supertable()  # Do not call process(); that would save outputs

    st = csn_instance.supertable.supertable
    print("Supertable shape:", st.shape)
    # Show a concise preview
    with pd.option_context('display.width', 160, 'display.max_columns', 20):
        print(st.head(10))


if __name__ == "__main__":
    main()


