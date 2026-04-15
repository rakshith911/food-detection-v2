"""
ByteTrack-style multi-object tracker for food video analysis.

Implements the core ByteTrack algorithm (Zhang et al., 2022) with a
constant-velocity Kalman filter for state prediction and two-stage
Hungarian matching.

Why this helps food videos:
- Kalman prediction keeps tracks alive during brief occlusions (e.g. hand
  reaches into frame) without waiting for re-detection.
- Two-stage matching recovers low-confidence or partially-visible detections
  instead of spawning a duplicate track ID.
- Track lifecycle (New → Tracked → Lost → Removed) prevents stale objects
  from polluting the nutrition results.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from enum import IntEnum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Kalman Filter
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    8-dimensional constant-velocity Kalman filter.

    State vector:  [cx, cy, a, h, vcx, vcy, va, vh]
      cx, cy  – bounding-box centre
      a       – aspect ratio (w / h)
      h       – height in pixels
      v*      – corresponding velocities

    Observation:   [cx, cy, a, h]
    """

    ndim = 4  # observation dimensions
    dt = 1.0  # time step (1 frame)

    def __init__(self):
        n = self.ndim
        # Transition matrix (constant-velocity model)
        self._F = np.eye(2 * n)
        for i in range(n):
            self._F[i, n + i] = self.dt

        # Observation matrix
        self._H = np.eye(n, 2 * n)

        # Process noise — higher velocity noise allows faster adaptation
        self._Q = np.diag([
            1e-2, 1e-2, 1e-4, 1e-2,   # position/shape
            5e-2, 5e-2, 5e-4, 5e-2,   # velocity
        ])

        # Observation noise
        self._R = np.diag([1e-1, 1e-1, 1e-3, 1e-1])

        # Initial covariance scale factors
        self._P_init_pos = 2.0
        self._P_init_vel = 10.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create track from first observation. Returns (mean, covariance)."""
        mean = np.zeros(2 * self.ndim)
        mean[:self.ndim] = measurement

        std = np.array([
            self._P_init_pos * measurement[3],
            self._P_init_pos * measurement[3],
            1e-2,
            self._P_init_pos * measurement[3],
            self._P_init_vel * measurement[3],
            self._P_init_vel * measurement[3],
            1e-4,
            self._P_init_vel * measurement[3],
        ])
        cov = np.diag(std ** 2)
        return mean, cov

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Advance state by one time step."""
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + self._Q
        return mean, cov

    def update(
        self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct state with new observation."""
        S = self._H @ cov @ self._H.T + self._R
        K = cov @ self._H.T @ np.linalg.inv(S)
        innovation = measurement - self._H @ mean
        mean = mean + K @ innovation
        cov = (np.eye(len(mean)) - K @ self._H) @ cov
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project state to observation space."""
        projected_mean = self._H @ mean
        projected_cov = self._H @ cov @ self._H.T + self._R
        return projected_mean, projected_cov


# ---------------------------------------------------------------------------
# Track state machine
# ---------------------------------------------------------------------------

class TrackState(IntEnum):
    New = 0       # just initialised, not yet confirmed
    Tracked = 1   # confirmed active track
    Lost = 2      # not matched for ≤ max_lost frames
    Removed = 3   # pruned from memory


# ---------------------------------------------------------------------------
# Single track
# ---------------------------------------------------------------------------

_kf = KalmanFilter()


class STrack:
    """
    Single food-item track.  Boxes stored as [x1, y1, x2, y2].
    """

    _next_id = 1

    def __init__(self, box: np.ndarray, label: str, score: float):
        self.track_id = STrack._next_id
        STrack._next_id += 1

        self.label = label
        self.score = score
        self.state = TrackState.New

        self.mean: Optional[np.ndarray] = None
        self.cov:  Optional[np.ndarray] = None
        self._initiate(box)

        self.frame_id = 0
        self.start_frame = 0
        self.frames_since_update = 0

    # ---- box conversion helpers ----

    @staticmethod
    def xyxy_to_cxcyah(box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        h = float(y2 - y1)
        a = float(x2 - x1) / max(h, 1e-6)
        return np.array([cx, cy, a, h], dtype=np.float32)

    @staticmethod
    def cxcyah_to_xyxy(obs: np.ndarray) -> np.ndarray:
        cx, cy, a, h = obs
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def _initiate(self, box: np.ndarray):
        obs = self.xyxy_to_cxcyah(box)
        self.mean, self.cov = _kf.initiate(obs)

    @property
    def box(self) -> np.ndarray:
        return self.cxcyah_to_xyxy(self.mean[:4])

    # ---- Kalman lifecycle ----

    def predict(self):
        self.mean, self.cov = _kf.predict(self.mean, self.cov)

    def update(self, box: np.ndarray, label: str, score: float):
        obs = self.xyxy_to_cxcyah(box)
        self.mean, self.cov = _kf.update(self.mean, self.cov, obs)
        self.label = label
        self.score = score
        self.state = TrackState.Tracked
        self.frames_since_update = 0

    def mark_lost(self):
        self.state = TrackState.Lost
        self.frames_since_update += 1

    def mark_removed(self):
        self.state = TrackState.Removed


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------

def _iou_matrix(tracks: List[STrack], boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between predicted track boxes and new detection boxes."""
    if not tracks or len(boxes) == 0:
        return np.zeros((len(tracks), len(boxes)), dtype=np.float32)

    t_boxes = np.array([t.box for t in tracks], dtype=np.float32)  # (T, 4)
    d_boxes = np.array(boxes, dtype=np.float32)                     # (D, 4)

    # Vectorised IoU
    inter_x1 = np.maximum(t_boxes[:, None, 0], d_boxes[None, :, 0])
    inter_y1 = np.maximum(t_boxes[:, None, 1], d_boxes[None, :, 1])
    inter_x2 = np.minimum(t_boxes[:, None, 2], d_boxes[None, :, 2])
    inter_y2 = np.minimum(t_boxes[:, None, 3], d_boxes[None, :, 3])

    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_t = (t_boxes[:, 2] - t_boxes[:, 0]) * (t_boxes[:, 3] - t_boxes[:, 1])
    area_d = (d_boxes[:, 2] - d_boxes[:, 0]) * (d_boxes[:, 3] - d_boxes[:, 1])
    union = area_t[:, None] + area_d[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def _hungarian_match(
    cost: np.ndarray, threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Run Hungarian algorithm on cost matrix, filter by threshold.
    Returns (matches, unmatched_rows, unmatched_cols).
    Cost is a *distance* matrix (1 - IoU).
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost)
    matches, unmatched_rows, unmatched_cols = [], [], []
    matched_rows, matched_cols = set(), set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= 1.0 - threshold:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_rows = [r for r in range(cost.shape[0]) if r not in matched_rows]
    unmatched_cols = [c for c in range(cost.shape[1]) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# BYTETracker
# ---------------------------------------------------------------------------

class BYTETracker:
    """
    ByteTrack multi-object tracker adapted for food video analysis.

    Key parameters
    --------------
    high_thresh   : IoU threshold for stage-1 matching (high-confidence detections)
    low_thresh    : IoU threshold for stage-2 matching (recovering lost tracks)
    max_lost      : frames a track can stay Lost before being removed
    min_hits      : frames needed before a New track becomes Tracked (confirmed)
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.3,
        max_lost: int = 5,
        min_hits: int = 1,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_lost = max_lost
        self.min_hits = min_hits

        self.tracked_tracks: List[STrack] = []
        self.lost_tracks:    List[STrack] = []
        self.removed_tracks: List[STrack] = []
        self.frame_id = 0

        # Reset the class-level ID counter each time a new tracker is created
        STrack._next_id = 1

    # ---- public API ----

    def update(
        self,
        boxes: np.ndarray,
        labels: List[str],
        scores: Optional[List[float]] = None,
    ) -> List[STrack]:
        """
        Process one detection frame.

        Parameters
        ----------
        boxes   : (N, 4) array in [x1, y1, x2, y2] format
        labels  : list of N label strings
        scores  : optional list of N confidence scores (default: 1.0 each)

        Returns
        -------
        List of currently active STrack objects (state Tracked).
        """
        self.frame_id += 1
        if scores is None:
            scores = [1.0] * len(boxes)

        boxes  = np.array(boxes,  dtype=np.float32) if len(boxes)  else np.zeros((0, 4), dtype=np.float32)
        scores = np.array(scores, dtype=np.float32) if len(scores) else np.zeros(0,       dtype=np.float32)

        # Split detections into high / low confidence
        high_mask = scores >= self.high_thresh
        low_mask  = (scores >= self.low_thresh) & ~high_mask

        high_boxes  = boxes[high_mask];   high_labels  = [labels[i] for i in np.where(high_mask)[0]]
        low_boxes   = boxes[low_mask];    low_labels   = [labels[i] for i in np.where(low_mask)[0]]
        high_scores = scores[high_mask];  low_scores   = scores[low_mask]

        # Predict new positions for all active tracks
        all_active = self.tracked_tracks + self.lost_tracks
        for t in all_active:
            t.predict()

        # ── Stage 1: match high-conf detections to all active tracks ──
        s1_matches, s1_unmatched_tracks, s1_unmatched_dets = [], list(range(len(all_active))), list(range(len(high_boxes)))
        if len(all_active) > 0 and len(high_boxes) > 0:
            iou = _iou_matrix(all_active, high_boxes)
            s1_matches, s1_unmatched_tracks, s1_unmatched_dets = _hungarian_match(1.0 - iou, self.high_thresh)
            for ti, di in s1_matches:
                all_active[ti].update(high_boxes[di], high_labels[di], float(high_scores[di]))

        # ── Stage 2: match low-conf detections to unmatched tracks ──
        unmatched_tracks_s1 = [all_active[i] for i in s1_unmatched_tracks]
        if len(unmatched_tracks_s1) > 0 and len(low_boxes) > 0:
            iou2 = _iou_matrix(unmatched_tracks_s1, low_boxes)
            s2_matches, s2_unmatched_tracks, _ = _hungarian_match(1.0 - iou2, self.low_thresh)
            for ti, di in s2_matches:
                unmatched_tracks_s1[ti].update(low_boxes[di], low_labels[di], float(low_scores[di]))
            still_unmatched = [unmatched_tracks_s1[i] for i in s2_unmatched_tracks]
        else:
            still_unmatched = unmatched_tracks_s1

        # Mark unmatched tracks as lost
        for t in still_unmatched:
            t.mark_lost()

        # Initialise new tracks from unmatched high-conf detections
        new_tracks = []
        for di in s1_unmatched_dets:
            t = STrack(high_boxes[di], high_labels[di], float(high_scores[di]))
            t.start_frame = self.frame_id
            t.frame_id    = self.frame_id
            new_tracks.append(t)

        # Promote New tracks to Tracked once they hit min_hits
        confirmed = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        confirmed += [t for t in new_tracks]

        # Prune lost tracks that have exceeded max_lost
        self.lost_tracks = [t for t in self.lost_tracks if t.frames_since_update <= self.max_lost]
        for t in self.lost_tracks:
            if t.frames_since_update > self.max_lost:
                t.mark_removed()

        self.removed_tracks += [t for t in self.lost_tracks if t.state == TrackState.Removed]
        self.lost_tracks     = [t for t in self.lost_tracks if t.state != TrackState.Removed]

        # Rebuild tracked_tracks: confirmed + new
        self.tracked_tracks = [t for t in confirmed if t.state in (TrackState.Tracked, TrackState.New)]
        # Move tracks that were just marked lost from tracked → lost pool
        newly_lost = [t for t in self.tracked_tracks if t.state == TrackState.Lost]
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state != TrackState.Lost]
        self.lost_tracks += newly_lost
        self.tracked_tracks += new_tracks

        return [t for t in self.tracked_tracks if t.state in (TrackState.Tracked, TrackState.New)]

    def get_active_tracks(self) -> List[STrack]:
        return [t for t in self.tracked_tracks if t.state in (TrackState.Tracked, TrackState.New)]
