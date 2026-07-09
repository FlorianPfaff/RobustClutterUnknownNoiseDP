"""Utilities for diagnosing and merging fragmented confirmed tracklets.

These helpers are primarily intended for simulation diagnostics. The source-ID
path uses simulation labels to expose fragmentation cleanly; the geometry path
is a label-free fallback for candidate real-data diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .tracklet import TentativeTracklet


@dataclass(frozen=True)
class MergedTracklet:
    """A group of confirmed tracklet fragments treated as one estimate."""

    group_id: str
    fragments: tuple[TentativeTracklet, ...]
    merge_reason: str

    def __post_init__(self) -> None:
        if not self.fragments:
            raise ValueError("fragments must not be empty")

    @property
    def num_fragments(self) -> int:
        return len(self.fragments)

    @property
    def first_scan(self) -> int:
        return min(fragment.first_scan for fragment in self.fragments)

    @property
    def last_scan(self) -> int:
        return max(fragment.last_scan for fragment in self.fragments)

    @property
    def duration(self) -> int:
        return self.last_scan - self.first_scan + 1

    @property
    def last_position(self) -> NDArray[np.float64]:
        return max(self.fragments, key=lambda fragment: fragment.last_scan).last_position

    @property
    def dominant_source_id(self) -> str | None:
        counts: dict[str, int] = {}
        for fragment in self.fragments:
            source_id = fragment.dominant_source_id
            if source_id is not None:
                counts[source_id] = counts.get(source_id, 0) + fragment.num_measurements
        if not counts:
            return None
        return max(counts, key=counts.__getitem__)


def merge_confirmed_tracklets(
    tracklets: Sequence[TentativeTracklet],
    max_gap: int = 2,
    max_distance: float = 1.0,
) -> tuple[MergedTracklet, ...]:
    """Merge confirmed tracklets by simulation source labels and geometry.

    Tracklets with the same non-null dominant source ID are merged first. The
    remaining unlabeled tracklets are greedily merged if their scan intervals are
    temporally adjacent and their endpoints are spatially close.
    """

    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")

    labeled: dict[str, list[TentativeTracklet]] = {}
    unlabeled: list[TentativeTracklet] = []
    for tracklet in tracklets:
        source_id = tracklet.dominant_source_id
        if source_id is None:
            unlabeled.append(tracklet)
        else:
            labeled.setdefault(source_id, []).append(tracklet)

    groups: list[MergedTracklet] = []
    for source_id, fragments in sorted(labeled.items()):
        ordered = tuple(sorted(fragments, key=lambda fragment: (fragment.first_scan, fragment.last_scan)))
        groups.append(MergedTracklet(group_id=f"source:{source_id}", fragments=ordered, merge_reason="source"))

    groups.extend(_merge_unlabeled_by_geometry(unlabeled, max_gap=max_gap, max_distance=max_distance))
    return tuple(sorted(groups, key=lambda group: (group.first_scan, group.group_id)))


def fragmentation_counts_by_truth(tracklets: Iterable[TentativeTracklet]) -> dict[str, int]:
    """Count confirmed tracklet fragments per simulation truth label."""

    counts: dict[str, int] = {}
    for tracklet in tracklets:
        source_id = tracklet.dominant_source_id
        if source_id is not None:
            counts[source_id] = counts.get(source_id, 0) + 1
    return counts


def mean_fragments_per_confirmed_truth(tracklets: Iterable[TentativeTracklet]) -> float:
    """Return the mean number of fragments for truth labels that were confirmed."""

    counts = fragmentation_counts_by_truth(tracklets)
    if not counts:
        return 0.0
    return float(sum(counts.values()) / len(counts))


def _merge_unlabeled_by_geometry(
    tracklets: Sequence[TentativeTracklet],
    max_gap: int,
    max_distance: float,
) -> tuple[MergedTracklet, ...]:
    groups: list[list[TentativeTracklet]] = []
    for tracklet in sorted(tracklets, key=lambda fragment: (fragment.first_scan, fragment.last_scan)):
        target_group = _find_geometry_group(tracklet, groups, max_gap=max_gap, max_distance=max_distance)
        if target_group is None:
            groups.append([tracklet])
        else:
            target_group.append(tracklet)

    merged = []
    for index, group in enumerate(groups, start=1):
        ordered = tuple(sorted(group, key=lambda fragment: (fragment.first_scan, fragment.last_scan)))
        merged.append(MergedTracklet(group_id=f"geometry:{index}", fragments=ordered, merge_reason="geometry"))
    return tuple(merged)


def _find_geometry_group(
    tracklet: TentativeTracklet,
    groups: Sequence[list[TentativeTracklet]],
    max_gap: int,
    max_distance: float,
) -> list[TentativeTracklet] | None:
    best_group: list[TentativeTracklet] | None = None
    best_distance = max_distance
    for group in groups:
        last = max(group, key=lambda fragment: fragment.last_scan)
        gap = tracklet.first_scan - last.last_scan
        if gap < 0 or gap > max_gap + 1:
            continue
        distance = float(np.linalg.norm(tracklet.first_position - last.last_position))
        if distance <= best_distance:
            best_distance = distance
            best_group = group
    return best_group
