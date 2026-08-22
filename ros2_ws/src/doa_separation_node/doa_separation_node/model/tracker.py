from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackerConfig:
    activity_threshold: float = 0.5
    gate_deg: float = 35.0
    confirm_chunks: int = 3
    coast_chunks: int = 6
    smooth_halflife_sec: float = 0.6
    max_tracks: int = 8


@dataclass
class Track:
    track_id: int
    direction: np.ndarray
    activity: float
    slot: int
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    age_sec: float = 0.0

    @property
    def azimuth_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.direction[1], self.direction[0])))

    @property
    def elevation_deg(self) -> float:
        return float(np.degrees(np.arcsin(np.clip(self.direction[2], -1.0, 1.0))))


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return np.degrees(np.arccos(np.clip(a @ b.T, -1.0, 1.0)))


class SourceTracker:

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def _new_id(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def update(self, direction: np.ndarray, activity: np.ndarray,
               dt: float) -> list[Track]:
        config = self.config
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / np.clip(
            np.linalg.norm(direction, axis=1, keepdims=True), 1e-9, None)
        activity = np.asarray(activity, dtype=np.float64)

        live = np.flatnonzero(activity >= config.activity_threshold)
        observed = direction[live]

        matched: dict[int, int] = {}
        if self.tracks and len(live):
            cost = angle_between(np.stack([t.direction for t in self.tracks]), observed)
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if cost[row, column] <= config.gate_deg:
                    matched[int(row)] = int(column)

        decay = 0.5 ** (dt / config.smooth_halflife_sec)
        for index, track in enumerate(self.tracks):
            track.age_sec += dt
            if index in matched:
                column = matched[index]
                blended = decay * track.direction + (1.0 - decay) * observed[column]
                track.direction = blended / max(np.linalg.norm(blended), 1e-9)
                track.activity = float(activity[live[column]])
                track.slot = int(live[column])
                track.hits += 1
                track.misses = 0
                if track.hits >= config.confirm_chunks:
                    track.confirmed = True
            else:
                track.activity = 0.0
                track.slot = -1
                track.hits = 0
                track.misses += 1

        taken = set(matched.values())
        for column, slot in enumerate(live):
            if column in taken or len(self.tracks) >= config.max_tracks:
                continue
            self.tracks.append(Track(track_id=self._new_id(),
                                     direction=observed[column],
                                     activity=float(activity[slot]),
                                     slot=int(slot), hits=1,
                                     confirmed=config.confirm_chunks <= 1))

        self.tracks = [t for t in self.tracks if t.misses < config.coast_chunks]
        return sorted((t for t in self.tracks if t.confirmed),
                      key=lambda t: -t.activity)

    def slot_to_track(self) -> dict[int, int]:
        return {t.slot: t.track_id for t in self.tracks
                if t.confirmed and t.slot >= 0}
