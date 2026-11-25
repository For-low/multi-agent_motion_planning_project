"""
planner.py — Multi-Variant Motion Planner for Local Testing & VPL Submission
=============================================================================
VARIANT controls which algorithm runs:
    "straight"   → Layer 1 only: straight-line bypass test
    "birrt"      → Layer 1 + BiRRT-Connect (baseline competitive)
    "halton"     → Layer 1 + Halton sampling + BiRRT
    "informed"   → Layer 1 + Halton + Informed Ellipsoid + BiRRT
    "full"       → All layers: informed BiRRT + priority shuffling + shortcutting

Set DEBUG_FLAG = False and VARIANT = "full" before VPL submission.
Set TEST_ENVS  = [1, 2, 3, 4] or None before VPL submission.
"""

import math
import time
import numpy as np
from motion_planning_env import UR5_JOINT_LOWER, UR5_JOINT_UPPER

# ============================================================================
# CONTROL FLAGS — change these for local testing
# ============================================================================
DEBUG_FLAG = True        # ← SET False BEFORE VPL SUBMISSION
VARIANT    = "full"      # ← "straight" | "birrt" | "halton" | "informed" | "full"
TEST_ENVS  = [1, 2, 3, 4]  # ← [1] for quick test, [1,2,3,4] for VPL

# ============================================================================
# PRECOMPUTED CONSTANTS
# ============================================================================
_LOWER = np.array(UR5_JOINT_LOWER, dtype=np.float64)
_UPPER = np.array(UR5_JOINT_UPPER, dtype=np.float64)
_RANGE = _UPPER - _LOWER   # shape (6,)
_NDIM  = 6

# ============================================================================
# HALTON SEQUENCE (Layer 2) — quasi-random low-discrepancy sampling
# ============================================================================
_HALTON_PRIMES = [2, 3, 5, 7, 11, 13]   # one prime per joint dimension

def _halton_single(index, base):
    """Compute one Halton value for given index and base."""
    result, f = 0.0, 1.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result

def _halton_sample(index):
    """Generate one 6D Halton sample mapped to joint space."""
    sample = np.array([_halton_single(index, p) for p in _HALTON_PRIMES])
    return _LOWER + sample * _RANGE

# Running Halton index (global counter so successive calls advance the sequence)
_halton_counter = [1]

def _next_halton():
    q = _halton_sample(_halton_counter[0])
    _halton_counter[0] += 1
    return q

def _reset_halton():
    _halton_counter[0] = 1

# ============================================================================
# SHARED HELPERS
# ============================================================================
def _weighted_dist(a, b):
    """Joint-range-normalized Euclidean distance in 6D."""
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)) / _RANGE))

def _random_config():
    return np.array([np.random.uniform(l, h)
                     for l, h in zip(UR5_JOINT_LOWER, UR5_JOINT_UPPER)])

def _steer(q_near, q_rand, step):
    """Move from q_near toward q_rand at most `step` in weighted space."""
    diff = q_rand - q_near
    wdiff = diff / _RANGE
    wdist = np.linalg.norm(wdiff)
    if wdist < 1e-9:
        return q_near.copy()
    if wdist <= step:
        return q_rand.copy()
    return q_near + (diff / wdist) * step

def _segment_clear(env, ri, q_from, q_to, n=8):
    """Return True if the straight segment is collision-free."""
    for t in np.linspace(0.0, 1.0, n):
        if env.check_collision_single_robot(ri, (q_from + t * (q_to - q_from)).tolist()):
            return False
    return True

def _extract_path(nodes, parents, idx):
    path = []
    while idx != -1:
        path.append(nodes[idx].copy())
        idx = parents[idx]
    path.reverse()
    return path

def _nearest_idx(nodes, q):
    dists = [_weighted_dist(n, q) for n in nodes]
    return int(np.argmin(dists))

# ============================================================================
# LAYER 5 — TIME-BOUNDED GREEDY SHORTCUTTING
# ============================================================================
def _shortcut(env, ri, path, max_time=0.3):
    """Random greedy shortcutting bounded by wall-clock time."""
    path = [np.array(p) for p in path]
    deadline = time.time() + max_time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(path) - 2 and time.time() < deadline:
            j = len(path) - 1
            while j > i + 1:
                n = max(6, int(_weighted_dist(path[i], path[j]) / 0.05))
                if _segment_clear(env, ri, path[i], path[j], n=n):
                    path = path[:i+1] + path[j:]
                    improved = True
                    break
                j -= 1
            i += 1
    return [p.tolist() for p in path]

# ============================================================================
# LAYER 3 — INFORMED ELLIPSOID SAMPLING
# ============================================================================
def _informed_sample(c_best, q_start, q_goal):
    """Sample from prolate hyperspheroid in weighted joint space."""
    if c_best == float('inf'):
        return _random_config()
    ws, wg = q_start / _RANGE, q_goal / _RANGE
    c_min = np.linalg.norm(wg - ws)
    if c_best <= c_min + 1e-9:
        return _random_config()
    center_w = (ws + wg) / 2.0
    L1    = c_best / 2.0
    L_rest = math.sqrt(max(c_best**2 - c_min**2, 1e-9)) / 2.0
    # Sample unit ball in 6D
    u = np.random.randn(_NDIM); u /= np.linalg.norm(u)
    r = np.random.uniform(0, 1) ** (1.0 / _NDIM)
    x_ball = r * u
    x_ball[0] *= L1; x_ball[1:] *= L_rest
    # Rotate first axis to align with (wg - ws)
    d = wg - ws
    if np.linalg.norm(d) < 1e-9:
        x_w = x_ball + center_w
    else:
        e1 = np.zeros(_NDIM); e1[0] = 1.0
        d_hat = d / np.linalg.norm(d)
        v = e1 - d_hat
        if np.linalg.norm(v) < 1e-9:
            M = np.eye(_NDIM)
        else:
            v /= np.linalg.norm(v)
            M = np.eye(_NDIM) - 2.0 * np.outer(v, v)
        x_w = M @ x_ball + center_w
    return np.clip(x_w * _RANGE, _LOWER, _UPPER)

# ============================================================================
# CORE BIRRT ENGINE
# ============================================================================
def _birrt(env, ri, start, goal,
           max_iter=5000, base_step=0.2, goal_bias=0.15,
           use_halton=False, use_informed=False):
    """
    Bidirectional RRT-Connect with optional Halton + Informed sampling.
    Returns best path found or None.
    """
    if use_halton:
        _reset_halton()

    nodes_a, parents_a = [start.copy()], [-1]
    nodes_b, parents_b = [goal.copy()],  [-1]

    c_best   = float('inf')
    best_path = None

    for it in range(max_iter):
        # Alternate which tree extends
        if it % 2 == 0:
            ext_n, ext_p, conn_n, conn_p = nodes_a, parents_a, nodes_b, parents_b
            toward_goal = True
        else:
            ext_n, ext_p, conn_n, conn_p = nodes_b, parents_b, nodes_a, parents_a
            toward_goal = False

        # --- SAMPLE ---
        rv = np.random.random()
        if rv < goal_bias:
            q_rand = conn_n[0].copy()           # bias toward other tree's root
        elif use_informed and c_best < float('inf') and rv < goal_bias + 0.45:
            q_rand = _informed_sample(c_best, start, goal)
        elif use_halton:
            q_rand = _next_halton()
        else:
            q_rand = _random_config()

        # --- EXTEND ---
        ni = _nearest_idx(ext_n, q_rand)
        q_near = ext_n[ni]
        wdist  = _weighted_dist(q_near, q_rand)
        step   = min(base_step, wdist) if wdist > 0 else base_step
        q_new  = _steer(q_near, q_rand, step)

        if env.check_collision_single_robot(ri, q_new.tolist()):
            continue
        n_seg = max(3, int(_weighted_dist(q_near, q_new) / 0.05))
        if not _segment_clear(env, ri, q_near, q_new, n=n_seg):
            continue

        ext_n.append(q_new.copy()); ext_p.append(ni)
        new_idx = len(ext_n) - 1

        # --- GREEDY CONNECT to other tree ---
        ci = _nearest_idx(conn_n, q_new)
        q_conn = conn_n[ci]
        q_try  = q_new.copy()
        conn_idx = None

        while True:
            wd = _weighted_dist(q_try, q_conn)
            if wd < 1e-6:
                conn_idx = ci; break
            step2 = min(base_step * 1.5, wd)
            q_next = _steer(q_try, q_conn, step2)
            if env.check_collision_single_robot(ri, q_next.tolist()):
                break
            n2 = max(4, int(_weighted_dist(q_try, q_next) / 0.05))
            if not _segment_clear(env, ri, q_try, q_next, n=n2):
                break
            conn_n.append(q_next.copy()); conn_p.append(ci)
            ci = len(conn_n) - 1
            q_try = q_next
            if _weighted_dist(q_try, q_conn) < step2 * 0.5:
                conn_idx = ci; break

        if conn_idx is None:
            continue

        # --- BUILD FULL PATH ---
        path_ext  = _extract_path(ext_n,  ext_p,  new_idx)
        path_conn = _extract_path(conn_n, conn_p, conn_idx)

        if toward_goal:
            full_path = path_ext + path_conn[::-1]
        else:
            full_path = path_conn + path_ext[::-1]

        cost = sum(_weighted_dist(full_path[k], full_path[k+1])
                   for k in range(len(full_path)-1))

        if cost < c_best:
            c_best    = cost
            best_path = [p.tolist() for p in full_path]
            if DEBUG_FLAG:
                print(f"    [BiRRT ri={ri}] iter={it} cost={c_best:.3f} nodes={len(full_path)}")

        # Early exit after 60% of budget if we have a solution
        if best_path and it > int(max_iter * 0.6):
            break

    return best_path

# ============================================================================
# LAYER 1 — STRAIGHT-LINE BYPASS
# ============================================================================
def _straight_line_plan(env, ri, start, goal):
    """Return [start, goal] if direct path is collision-free, else None."""
    n = max(10, int(_weighted_dist(start, goal) / 0.03))
    if _segment_clear(env, ri, start, goal, n=n):
        return [start.tolist(), goal.tolist()]
    return None

# ============================================================================
# LAYER 4 — PRIORITIZED MULTI-ROBOT PLANNING WITH DETERMINISTIC RE-SCHEDULING
# ============================================================================
def _plan_single_robot(env, ri, start, goal, variant, time_limit=30.0):
    """Plan for one robot using the selected variant."""
    s, g = np.array(start), np.array(goal)
    t0 = time.time()

    # Layer 1: always try straight line first
    path = _straight_line_plan(env, ri, s, g)
    if path is not None:
        if DEBUG_FLAG:
            print(f"  [ri={ri}] Straight-line path found instantly!")
        return path

    if time.time() - t0 > time_limit:
        return None

    if variant == "straight":
        return None   # straight only, no RRT fallback

    use_h = variant in ("halton", "informed", "full")
    use_i = variant in ("informed", "full")

    path = _birrt(env, ri, s, g,
                  max_iter=5000, base_step=0.2, goal_bias=0.15,
                  use_halton=use_h, use_informed=use_i)

    if path and variant == "full":
        elapsed = time.time() - t0
        shortcut_budget = max(0.05, min(0.4, time_limit - elapsed - 0.1))
        path = _shortcut(env, ri, path, max_time=shortcut_budget)

    return path


class StudentPlanner:
    def __init__(self):
        pass

    def plan(self, env, start_configs, goal_configs):
        """
        Prioritized multi-robot planning with deterministic re-scheduling
        (Andreychuk & Yakovlev, 2018 — Layer 4).
        """
        n = env.num_robots
        base_order = list(range(n))

        # Build priority orders to try:
        # 1st: natural order [0,1,2,...,n-1]
        # 2nd onwards: if robot k fails due to robot i, move k above i
        orders_to_try = [base_order.copy()]
        if n > 1:
            for start_robot in range(1, n):
                # Put robot `start_robot` first, others follow
                new_order = [start_robot] + [r for r in base_order if r != start_robot]
                orders_to_try.append(new_order)

        per_robot_time = 25.0 / max(n, 1)   # share total time budget

        for attempt, order in enumerate(orders_to_try):
            if DEBUG_FLAG:
                print(f"\n[Plan attempt {attempt+1}] Order: {order}")

            paths = [None] * n
            failed_robot = None

            for r in order:
                t0 = time.time()
                path = _plan_single_robot(
                    env, r,
                    start_configs[r], goal_configs[r],
                    variant=VARIANT, time_limit=per_robot_time
                )
                elapsed = time.time() - t0

                if path is None:
                    if DEBUG_FLAG:
                        print(f"  [ri={r}] FAILED after {elapsed:.3f}s")
                    failed_robot = r
                    break

                paths[r] = path
                if DEBUG_FLAG:
                    plen = sum(
                        np.linalg.norm(np.array(path[k+1]) - np.array(path[k]))
                        for k in range(len(path)-1)
                    )
                    print(f"  [ri={r}] OK in {elapsed:.4f}s  waypoints={len(path)}  len={plen:.2f}")

            if failed_robot is None:
                return paths   # all robots planned successfully

            if DEBUG_FLAG:
                print(f"  → Retry with robot {failed_robot} prioritized first")

        return None   # all orderings exhausted


def get_planner():
    return StudentPlanner()
