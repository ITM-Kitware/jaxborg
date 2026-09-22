import json

import pytest

from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
from scripts.eval import eval_cross_play_fsm


@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
@pytest.mark.parametrize("legacy_seeds", [False, True])
def test_checkpoint_replay_matches_algorithm_and_episode_protocol(tmp_path, monkeypatch, algorithm, legacy_seeds):
    run = tmp_path / f"{algorithm}_jax" / "run"
    run.mkdir(parents=True)
    checkpoint = run / "checkpoint_40.safetensors"
    checkpoint.touch()
    topology = tmp_path / "topology.npz"
    topology.touch()
    monkeypatch.setattr(eval_cross_play_fsm, "read_sidecar", lambda _: {})
    monkeypatch.setattr(eval_cross_play_fsm, "eval_variant", lambda _: CC4_STOCK)
    cell = {
        "blue_step": 40,
        "blue_checkpoint": f"/original/{algorithm}_jax/run/checkpoint_40.safetensors",
        "variant": CC4_STOCK.name,
        "topology_paths": [str(topology)],
        "topology_sampling": "exhaustive",
        "seeds": [10, 11],
        "episodes_per_seed": 2,
        "per_episode_seeds": [10, 11, 11, 12] if legacy_seeds else [20, 21, 22, 23],
        "stochastic": True,
    }
    source = tmp_path / "cross_play.jsonl"
    summary = {"eval_name": "cross_play_summary", "steps": [40], "eval_id": "saved"}
    source.write_text(json.dumps(cell) + "\n" + json.dumps(summary) + "\n")
    if legacy_seeds:
        with pytest.raises(ValueError, match="different episode seed or topology protocol"):
            list(eval_cross_play_fsm.matched_checkpoints(source, tmp_path, tmp_path))
    else:
        [(model, _, _, _, signature)] = eval_cross_play_fsm.matched_checkpoints(source, tmp_path, tmp_path)
        assert model == checkpoint
        assert signature["per_episode_seeds"] == [20, 21, 22, 23]
