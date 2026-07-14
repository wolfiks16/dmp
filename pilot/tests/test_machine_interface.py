import pytest


def test_load_pmsm_config_valid():
    pytest.importorskip("gmsh")
    from pilot.machine_config import load_machine_run

    run = load_machine_run("pilot/configs/pmsm.toml")
    assert run.problem.validate() == []          # согласованная постановка
    assert run.problem.scenario.value == "S3"
    assert run.problem.has_current
    assert run.material_name == "NdFeB"


def test_end_to_end_s1_writes_outputs(tmp_path):
    pytest.importorskip("gmsh")
    from pilot.machine_run import main

    cfg = tmp_path / "s1.toml"
    cfg.write_text(
        "[geometry]\nmesh_size = 0.005\n"
        "[magnet]\nmaterial = 'smco'\n"
        "[winding]\nturns_per_slot = 30\n"
        "[regime]\nscenario = 'S1'\n"
        f"[output]\ndir = '{tmp_path.as_posix()}'\n",
        encoding="utf-8",
    )
    rc = main([str(cfg)])
    assert rc == 0
    assert (tmp_path / "machine_report.txt").exists()
    assert (tmp_path / "machine_risk.png").exists()
    assert (tmp_path / "machine_operating_point.png").exists()


def test_invalid_config_rejected(tmp_path):
    pytest.importorskip("gmsh")
    from pilot.machine_run import main

    cfg = tmp_path / "bad.toml"
    # S1 с T≠20 — несогласованная постановка, должна быть отвергнута (rc=1).
    cfg.write_text(
        "[geometry]\nmesh_size = 0.005\n"
        "[regime]\nscenario = 'S1'\nT = 150.0\n"
        f"[output]\ndir = '{tmp_path.as_posix()}'\n",
        encoding="utf-8",
    )
    assert main([str(cfg)]) == 1
