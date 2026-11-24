"""Validate completed replay cases and summarize paired Blue outcomes."""

import argparse
import json
from collections import defaultdict
from itertools import product
from math import isfinite
from pathlib import Path
from statistics import mean

PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text())
CONDITIONS = ("one", "hundred")
LEARNED = tuple(f"learned-{condition}" for condition in CONDITIONS)
SCRIPTED = tuple(PROTOCOL["scripted_reds"])
PANELS = ("familiar", "held-out")


def read(path):
    return json.loads(path.read_text())


def to_episodes(layout, seeds, rewards, cia):
    return [
        {"layout_seed": layout, "episode_seed": int(seed), "reward": reward, "targeted_cia": metric}
        for seed, reward, metric in zip(seeds, rewards, cia, strict=True)
    ]


def archived_familiar(learned_dir, scripted_dir):
    layout = PROTOCOL["familiar_topology_seed"]
    fingerprint = PROTOCOL["familiar_topology_fingerprint"]
    for seed, condition in product(PROTOCOL["training_seeds"], CONDITIONS):
        red_seed = next(other for other in PROTOCOL["training_seeds"] if other != seed)
        for red_condition in CONDITIONS:
            path = learned_dir / f"blue-{condition}-{seed}_red-{red_condition}-{red_seed}.json"
            row = read(path)
            if row["layout_fingerprint"] != fingerprint:
                raise ValueError(f"Archived learned layout fingerprint mismatch in {path}")
            for team, expected_seed in (("blue", seed), ("red", red_seed)):
                policy = row["policies"][team]
                expected_condition = condition if team == "blue" else red_condition
                expected_recipe = PROTOCOL["checkpoints"][expected_condition][str(expected_seed)]
                if (
                    policy["train_seed"] != expected_seed
                    or policy["bundle_provenance"]["total_steps"] != PROTOCOL["checkpoint_step"]
                    or Path(policy["path"]).parent.name != expected_recipe["relative_directory"]
                ):
                    raise ValueError(f"Archived {team} policy provenance mismatch in {path}")
            scores = row["episode_returns"]
            yield (
                ("familiar", f"learned-{red_condition}", seed, condition),
                to_episodes(layout, scores.keys(), scores.values(), [None] * len(scores)),
            )
        for red in SCRIPTED:
            path = scripted_dir / f"{red}-{condition}-{seed}.json"
            row = read(path)
            if (
                set(row["per_episode_topology_fingerprints"]) != {fingerprint}
                or row["train_total_steps"] != PROTOCOL["checkpoint_step"]
                or row["blue_training_seed"] != seed
                or row["eval_red"] != red
            ):
                raise ValueError(f"Archived scripted provenance mismatch in {path}")
            target = red[-1] if red.startswith("cia_") else None
            cia = [episode.get(target) for episode in row["per_episode_cia"]]
            yield (
                ("familiar", red, seed, condition),
                to_episodes(layout, row["per_episode_seeds"], row["per_episode"], cia),
            )


def replay_rows(root, panel):
    for path in sorted(root.glob(f"{panel}-task-*/[oh]*-*.json")):
        row = read(path)
        if row["panel"] != panel:
            raise ValueError(f"Wrong panel in {path}")
        red_seed = row["red_training_seed"]
        if row["opponent"] in LEARNED:
            if red_seed not in PROTOCOL["training_seeds"] or red_seed == row["training_seed"]:
                raise ValueError(f"Wrong held-out Red seed in {path}")
        elif red_seed is not None:
            raise ValueError(f"Scripted Red has a training seed in {path}")
        yield (panel, row["opponent"], row["training_seed"], row["condition"]), row["episodes"]


def validate_rows(rows, layouts):
    expected = set(product(PANELS, (*LEARNED, *SCRIPTED), PROTOCOL["training_seeds"], CONDITIONS))
    validated = {}
    for key, episodes in rows:
        if key not in expected or key in validated:
            raise ValueError(f"Unexpected or repeated checkpoint evaluation {key}")
        panel, opponent, _, _ = key
        expected_cases = set(product(layouts[panel], PROTOCOL["episode_seeds"]))
        by_case = {(row["layout_seed"], row["episode_seed"]): row for row in episodes}
        if set(by_case) != expected_cases or len(by_case) != len(episodes):
            raise ValueError(f"Incomplete or repeated cases in {key}")
        for row in episodes:
            cia = row["targeted_cia"]
            if not isfinite(row["reward"]) or (opponent.startswith("cia_") and (cia is None or not isfinite(cia))):
                raise ValueError(f"Missing or invalid score in {key}")
        validated[key] = by_case
    if missing := expected - validated.keys():
        raise ValueError(f"Missing checkpoint evaluations: {sorted(missing)}")
    return validated


def summarize(rows, layouts):
    scores = {}
    maps = defaultdict(list)
    for panel, opponent, seed in product(PANELS, (*LEARNED, *SCRIPTED), PROTOCOL["training_seeds"]):
        one = rows[panel, opponent, seed, "one"]
        hundred = rows[panel, opponent, seed, "hundred"]
        for case, old in one.items():
            new = hundred[case]
            if old.get("role_map_id") != new.get("role_map_id"):
                raise ValueError(f"Mismatched CIA role map for {panel}, {opponent}, {seed}, {case}")
        for layout in layouts[panel]:
            pairs = [(old, hundred[case]) for case, old in one.items() if case[0] == layout]
            maps[panel, opponent, seed].append(
                {
                    "layout_seed": layout,
                    "one": mean(old["reward"] for old, _ in pairs),
                    "hundred": mean(new["reward"] for _, new in pairs),
                    "reward_gap": mean(new["reward"] - old["reward"] for old, new in pairs),
                    "cia_gap": mean(new["targeted_cia"] - old["targeted_cia"] for old, new in pairs)
                    if opponent.startswith("cia_")
                    else None,
                }
            )
        group = maps[panel, opponent, seed]
        scores[panel, opponent, seed] = {key: mean(m[key] for m in group) for key in ("one", "hundred", "reward_gap")}
        if opponent.startswith("cia_"):
            scores[panel, opponent, seed]["cia_gap"] = mean(m["cia_gap"] for m in group)
    return scores, maps


def report(scores, layouts):
    lines = [
        "Positive gaps favor the 100-layout Blue. Each panel weights its layouts equally.",
        "",
        "| Blue training seed | Frozen opponent | Map 0 reward gap | Held-out reward gap | Gap change |",
        "|---|---|---:|---:|---:|",
    ]
    for seed, red in product(PROTOCOL["training_seeds"], (*LEARNED, *SCRIPTED)):
        old = scores["familiar", red, seed]["reward_gap"]
        new = scores["held-out", red, seed]["reward_gap"]
        lines.append(f"| {seed} | {red} | {old:+.2f} | {new:+.2f} | {new - old:+.2f} |")
    lines += [
        "",
        "| Blue training seed | Opponent family | Map 0 reward gap | Held-out reward gap | Gap change |",
        "|---|---|---:|---:|---:|",
    ]
    for seed in PROTOCOL["training_seeds"]:
        for family, reds in (("Learned", LEARNED), ("CIA-focused", SCRIPTED[:3]), ("H-MARL scripted", SCRIPTED[3:])):
            old, new = (mean(scores[panel, red, seed]["reward_gap"] for red in reds) for panel in PANELS)
            lines.append(f"| {seed} | {family} | {old:+.2f} | {new:+.2f} | {new - old:+.2f} |")
    lines += [
        "",
        "| Blue training seed | Opponent | Map 0 targeted CIA gap | Held-out targeted CIA gap |",
        "|---|---|---:|---:|",
    ]
    for seed, red in product(PROTOCOL["training_seeds"], SCRIPTED[:3]):
        old, new = (scores[panel, red, seed]["cia_gap"] for panel in PANELS)
        lines.append(f"| {seed} | {red} | {old:+.3f} | {new:+.3f} |")
    lines += ["", f"Held-out layout seeds: {', '.join(map(str, layouts['held-out']))}.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_root", type=Path)
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--archived-learned", type=Path)
    parser.add_argument("--archived-scripted", type=Path)
    args = parser.parse_args()
    if bool(args.archived_learned) != bool(args.archived_scripted):
        parser.error("Supply both archived familiar-map directories or neither")
    selected = read(args.replay_root / "layouts.json")["selected"]
    layouts = {panel: [layout["seed"] for layout in selected[panel]] for panel in PANELS}
    familiar = (
        archived_familiar(args.archived_learned, args.archived_scripted)
        if args.archived_learned
        else replay_rows(args.replay_root, "familiar")
    )
    rows = validate_rows((*familiar, *replay_rows(args.replay_root, "held-out")), layouts)
    scores, maps = summarize(rows, layouts)
    args.report_root.mkdir(parents=True, exist_ok=False)
    (args.report_root / "report.md").write_text(report(scores, layouts))
    (args.report_root / "map-means.json").write_text(
        json.dumps(
            [
                {"panel": panel, "opponent": opponent, "blue_training_seed": seed, **row}
                for (panel, opponent, seed), group in maps.items()
                for row in group
            ],
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
