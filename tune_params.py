"""
tune_params.py — Grid search over BiRRT hyperparameters
Usage: python tune_params.py --env 1 --pairs 5
Tests different max_iter, base_step, goal_bias combinations.
"""
import time, argparse
import numpy as np
from motion_planning_env import MotionPlanningEnv
import planner as pm

PARAM_GRID = {
    "max_iter":  [2000, 4000, 6000],
    "base_step": [0.1,  0.2,  0.3],
    "goal_bias": [0.05, 0.10, 0.15, 0.20],
}

def test_params(env_id, num_pairs, seed, max_iter, base_step, goal_bias):
    np.random.seed(seed)
    pm.DEBUG_FLAG = False
    env = MotionPlanningEnv(env_id)
    times, lens, solved = [], [], 0
    for _ in range(num_pairs):
        s_cfgs, g_cfgs = env.sample_start_goal()
        s = np.array(s_cfgs[0]); g = np.array(g_cfgs[0])
        # straight-line first
        path = pm._straight_line_plan(env, 0, s, g)
        t0 = time.time()
        if path is None:
            path = pm._birrt(env, 0, s, g,
                             max_iter=max_iter, base_step=base_step,
                             goal_bias=goal_bias,
                             use_halton=True, use_informed=True)
        elapsed = time.time() - t0
        times.append(elapsed)
        if path:
            solved += 1
            lens.append(sum(np.linalg.norm(np.array(path[j+1]) - np.array(path[j]))
                            for j in range(len(path)-1)))
    pm.DEBUG_FLAG = True
    return {"solved": solved, "avg_time": np.mean(times),
            "max_time": np.max(times), "avg_len": np.mean(lens) if lens else float('nan')}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",   type=int, default=1)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    print(f"\n  PARAM TUNING  ENV={args.env}  pairs={args.pairs}")
    print(f"\n  {'iter':>6} {'step':>6} {'bias':>6}  {'Solved':>7}  {'AvgT':>8}  {'MaxT':>8}  {'AvgLen':>8}")
    print(f"  {'─'*60}")

    all_results = []
    for mi in PARAM_GRID["max_iter"]:
        for bs in PARAM_GRID["base_step"]:
            for gb in PARAM_GRID["goal_bias"]:
                r = test_params(args.env, args.pairs, args.seed, mi, bs, gb)
                all_results.append({**r, "max_iter": mi, "base_step": bs, "goal_bias": gb})
                print(f"  {mi:>6} {bs:>6.2f} {gb:>6.2f}  "
                      f"{r['solved']}/{args.pairs:>4}  "
                      f"{r['avg_time']:>8.4f}s  {r['max_time']:>8.4f}s  {r['avg_len']:>8.2f}")

    full = [r for r in all_results if r["solved"] == args.pairs]
    if full:
        best_t = min(full, key=lambda x: x["avg_time"])
        best_l = min(full, key=lambda x: x["avg_len"])
        print(f"\n  🏆 Best time:   iter={best_t['max_iter']} step={best_t['base_step']} bias={best_t['goal_bias']}  → {best_t['avg_time']:.4f}s")
        print(f"  🏆 Best length: iter={best_l['max_iter']} step={best_l['base_step']} bias={best_l['goal_bias']}  → {best_l['avg_len']:.2f}")
    else:
        print("\n  ⚠ No param combination solved all pairs. Increase max_iter.")
