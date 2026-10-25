import os

"""Make candidate-overlap-grouped development/validation/test partitions.

This is only a provisional evaluation split. Source titles and medleys do not
establish song identity; acoustic matching can miss covers and shared songs.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    tracks = [
        json.loads(line)
        for line in (ROOT / "data/manifests/tracks.jsonl").read_text().splitlines()
    ]
    review = {
        row["track_id"]: row
        for row in csv.DictReader((ROOT / "reports/song_identity_review.csv").open())
    }
    parent = {row["id"]: row["id"] for row in tracks if row["eligible"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def merge(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    overlap = list(
        csv.DictReader((ROOT / "reports/audio_overlap_candidates.csv").open())
    )
    for pair in overlap:
        if float(pair["score"]) >= 0.95:
            merge(pair["track_a"], pair["track_b"])
    # Keep all eight tracks from the first debug run in training, so future
    # held-out measurements do not accidentally include an exposed recording.
    old_run = json.loads((ROOT / "outputs/tiny_e2/run.json").read_text())
    exposed = {pair_id.split("_")[0] for pair_id in old_run["pair_ids"]}
    grouped = {track: find(track) for track in parent}
    splits = {}
    for group in set(grouped.values()):
        members = {track for track, root in grouped.items() if root == group}
        if members & exposed:
            split = "train"
        else:
            score = (
                int(
                    hashlib.sha256(("IE643-split-v1:" + group).encode()).hexdigest()[
                        :8
                    ],
                    16,
                )
                / 0xFFFFFFFF
            )
            split = (
                "test" if score < 0.15 else "validation" if score < 0.30 else "train"
            )
        splits[group] = split
    eligible_rows = []
    for row in tracks:
        updated = dict(row)
        if row["eligible"]:
            track_id = row["id"]
            updated["candidate_recording_group"] = grouped[track_id]
            updated["split"] = splits[grouped[track_id]]
            updated["split_status"] = (
                "provisional_recording_grouped_song_identity_unverified"
            )
            updated["possible_compilation"] = (
                review[track_id]["possible_compilation"] == "True"
            )
            eligible_rows.append(updated)
        else:
            updated["split"] = "excluded"
        row.clear()
        row.update(updated)
    assert not any(
        row["id"] in exposed and row["split"] != "train" for row in eligible_rows
    )
    by_group = {}
    for row in eligible_rows:
        by_group.setdefault(row["candidate_recording_group"], set()).add(row["split"])
    assert all(len(values) == 1 for values in by_group.values())
    out = ROOT / "data/manifests/tracks_provisional_split.jsonl"
    out.write_text("".join(json.dumps(row) + "\n" for row in tracks))
    original_pairs = [
        json.loads(line)
        for line in (ROOT / "data/manifests/synthetic_pairs.jsonl")
        .read_text()
        .splitlines()
    ]
    by_id = {row["id"]: row for row in eligible_rows}
    for pair in original_pairs:
        track = by_id[pair["track_id"]]
        pair["split"] = track["split"]
        pair["split_status"] = track["split_status"]
        pair["candidate_recording_group"] = track["candidate_recording_group"]
        pair["possible_compilation"] = track["possible_compilation"]
    (ROOT / "data/manifests/synthetic_pairs_provisional_split.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in original_pairs)
    )
    report = {
        "status": "provisional",
        "eligible_tracks": len(eligible_rows),
        "track_counts": dict(Counter(row["split"] for row in eligible_rows)),
        "pair_counts": dict(Counter(row["split"] for row in original_pairs)),
        "overlap_groups_with_multiple_tracks": sum(
            len([x for x in eligible_rows if x["candidate_recording_group"] == g]) > 1
            for g in by_group
        ),
        "known_debug_exposures_forced_to_train": len(exposed),
        "split_leakage_across_detected_groups": 0,
        "warning": "Track split is provisional. Medleys, covers and undetected recording overlap can still leak songs across groups. Do not call this a verified song-disjoint benchmark.",
    }
    (ROOT / "reports/provisional_split.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
