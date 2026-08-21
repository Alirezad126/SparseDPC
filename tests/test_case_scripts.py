import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_case_studies_have_only_independent_stage_scripts():
    expected = {
        "DoubleIntegrator": {
            "system_id.py",
            "train_sd_dpc.py",
            "train_nn_dpc.py",
            "unconstrained_adaptation.py",
            "safe_adaptation.py",
            "evaluation.py",
        },
        "TwoTank": {
            "system_id.py",
            "train_sd_dpc.py",
            "train_nn_dpc.py",
            "unconstrained_adaptation.py",
            "safe_adaptation.py",
            "evaluation.py",
        },
        "VanDerPol": {
            "system_id.py",
            "train_sd_dpc.py",
            "train_nn_dpc.py",
            "evaluation.py",
        },
    }
    for case_name, filenames in expected.items():
        actual = {path.name for path in (SCRIPTS / case_name).glob("*.py")}
        assert actual == filenames

    assert not (ROOT / "Relative_Degree_One" / "scripts").exists()
    assert not (SCRIPTS / "run_all_experiments.py").exists()
    assert not (SCRIPTS / "retrain_all_policies.py").exists()


def test_all_case_scripts_parse_and_do_not_accept_a_system_selector():
    for script in SCRIPTS.glob("*/*.py"):
        source = script.read_text()
        ast.parse(source, filename=str(script))
        assert "--system" not in source
        assert "run_all" not in source
