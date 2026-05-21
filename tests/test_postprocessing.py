"""Tests for simulation_postprocessing and experiment_postprocessing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

MWE = Path(__file__).parent.parent / "mwe_data"
EXP = Path(__file__).parent.parent / "experiment_workflow_aps_28Feb"


@pytest.fixture
def sim_results():
    if not (MWE / "out.csv").exists():
        pytest.skip("mwe_data/out.csv not found")
    from graintrace.simulation_postprocessing import FieldFileNaming, SimulationResults

    naming = FieldFileNaming(
        prefix="out_element_centroid",
        index_width=4,
        sep="_",
        suffix=".csv",
    )
    return SimulationResults(
        block_csv=MWE / "out.csv",
        field_dir=MWE / "grid_out",
        field_naming=naming,
    )


class TestFieldFileNaming:
    def test_defaults(self):
        from graintrace.simulation_postprocessing import FieldFileNaming

        fn = FieldFileNaming(prefix="foo")
        assert fn.prefix == "foo"
        assert fn.index_width is None
        assert fn.sep == "_"
        assert fn.suffix == ".csv"

    def test_custom(self):
        from graintrace.simulation_postprocessing import FieldFileNaming

        fn = FieldFileNaming(prefix="bar", index_width=4, sep="-", suffix=".dat")
        assert fn.index_width == 4
        assert fn.sep == "-"
        assert fn.suffix == ".dat"


class TestSimulationResults:
    def test_loads(self, sim_results):
        assert sim_results.n_steps > 0
        assert len(sim_results.grain_ids) > 0

    def test_n_steps_positive(self, sim_results):
        assert sim_results.n_steps > 0

    def test_get_tensor_block_scalar(self, sim_results):
        gid = sim_results.grain_ids[0]
        data = sim_results.get_tensor_block("volume", order=0, sample="time", grain_id=gid)
        assert data.shape == (sim_results.n_steps, 1)

    def test_get_tensor_block_tensor(self, sim_results):
        gid = sim_results.grain_ids[0]
        data = sim_results.get_tensor_block("ee", order=2, sample="time", grain_id=gid)
        assert data.shape == (sim_results.n_steps, 9)

    def test_get_tensor_block_by_id(self, sim_results):
        data = sim_results.get_tensor_block(
            "ee", order=2, sample="id", block_id=0
        )
        assert data.ndim == 2
        assert data.shape[1] == 9

    def test_get_tensor_block_returns_comp_names(self, sim_results):
        gid = sim_results.grain_ids[0]
        data, names = sim_results.get_tensor_block(
            "volume", order=0, sample="time", grain_id=gid, return_comp_names=True
        )
        assert isinstance(names, list)
        assert len(names) == 1

    def test_invalid_sample_raises(self, sim_results):
        with pytest.raises(ValueError, match="sample"):
            sim_results.get_tensor_block("volume", order=0, sample="bad")

    def test_invalid_order_raises(self, sim_results):
        gid = sim_results.grain_ids[0]
        with pytest.raises(ValueError, match="order"):
            sim_results.get_tensor_block("volume", order=3, sample="time", grain_id=gid)


@pytest.fixture
def exp_results():
    if not EXP.exists():
        pytest.skip("experiment_workflow_aps_28Feb/ not found")
    from graintrace.experiment_postprocessing import ExperimentResults, FieldFileNaming

    naming = FieldFileNaming(prefix="stitched_output", index_width=1, sep="", suffix=".csv")
    return ExperimentResults(exp_dir=EXP, exp_naming=naming)


class TestExperimentResults:
    def test_loads(self, exp_results):
        assert exp_results.n_steps > 0

    def test_grain_ids_populated(self, exp_results):
        assert len(exp_results.grain_ids) > 0

    def test_get_tensor_block_by_id(self, exp_results):
        data = exp_results.get_tensor_block(
            "eKen", order=2, sample="id", block_id=0
        )
        assert data.ndim == 2
        assert data.shape[1] == 9

    def test_get_tensor_block_eul_vector(self, exp_results):
        data = exp_results.get_tensor_block(
            "Eul", order=1, sample="id", block_id=0
        )
        assert data.shape[1] == 3

    def test_get_tensor_element_raises(self, exp_results):
        with pytest.raises(NotImplementedError):
            exp_results.get_tensor_element()
