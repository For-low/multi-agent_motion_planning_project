"""
compare_variants.py — Side-by-side comparison of all 5 algorithm variants
Usage: python compare_variants.py --env 1 --pairs 5
IMPORTANT: This modifies planner.VARIANT at runtime — safe because it's just a variable.
"""
import time, argparse, importlib
import numpy as np
from motion_planning_env import MotionPlanningEnv
import planner as planner_module

VARIANTS = ["straight", "birrt", "halton", "informed", "full"]

def test_variant(variant_name, env_id, num_pairs, seed):
    np.random.seed(seed)
    planner_module.VARIANT = variant_name
    planner_module.DEBUG_FLAG = False   # suppress output during comparison
    p = planner_module.get_planner()
    env = MotionPlanningEnv(env_id)

    times, lens, solved = [], [], 0
    for _ in range(num_pairs):
        s, g = env.sample_start_goal()
        t0 = time.time()
        paths = p.plan(env, s, g)
        elapsed = time.time() - t0
        times.append(elapsed)
        if paths:
            solved += 1
            total_len = sum(
                sum(np.linalg.norm(np.array(path[j+1]) - np.array(path[j]))
                    for j in range(len(path)-1))
                for path in paths
            )
            lens.append(total_len)

    return {
        "variant":  variant_name,
        "solved":   solved,
        "total":    num_pairs,
        "avg_time": np.mean(times),
        "max_time": np.max(times),
        "avg_len":  np.mean(lens) if lens else float('nan'),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",   type=int, default=1)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    print(f"\n  VARIANT COMPARISON  ENV={args.env}  pairs={args.pairs}  seed={args.seed}")
    print(f"\n  {'Variant':<12} {'Solved':>8}  {'AvgTime':>9}  {'MaxTime':>9}  {'AvgLen':>9}")
    print(f"  {'─'*55}")

    results = []
    for v in VARIANTS:
        print(f"  Testing '{v}'...", end="\r")
        r = test_variant(v, args.env, args.pairs, args.seed)
        results.append(r)
        print(f"  {r['variant']:<12} {r['solved']}/{r['total']:>5}  "
              f"{r['avg_time']:>9.4f}s  {r['max_time']:>9.4f}s  {r['avg_len']:>9.2f}")

    best_t = min(results, key=lambda x: x['avg_time'] if x['solved'] == x['total'] else 999)
    best_l = min(results, key=lambda x: x['avg_len']  if x['solved'] == x['total'] else 999)
    print(f"\n  🏆 Fastest (40% grade):   {best_t['variant']}  ({best_t['avg_time']:.4f}s avg)")
    print(f"  🏆 Shortest (20% grade):  {best_l['variant']}  ({best_l['avg_len']:.2f} avg len)")

    # Restore defaults
    planner_module.DEBUG_FLAG = True
    planner_module.VARIANT = "full"
