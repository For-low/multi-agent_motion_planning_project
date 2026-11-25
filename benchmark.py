"""
benchmark.py — Local benchmark for all environments
Usage:
    python benchmark.py --env 1 --pairs 10
    python benchmark.py --all                  # runs all 4 envs, 10 pairs each
    python benchmark.py --env 1 --pairs 5 --seed 7
"""
import time, argparse
import numpy as np
from motion_planning_env import MotionPlanningEnv
from planner import get_planner, VARIANT

def benchmark(env_id, num_pairs=10, seed=42):
    np.random.seed(seed)
    planner = get_planner()
    env = MotionPlanningEnv(env_id)

    times, path_lengths, solved = [], [], 0

    print(f"\n{'─'*60}")
    print(f"  ENV {env_id}  |  VARIANT={VARIANT}  |  pairs={num_pairs}  |  seed={seed}")
    print(f"{'─'*60}")
    print(f"  {'Pair':<5} {'Time':>8}  {'PathLen':>9}  {'Waypoints':<20}  {'Status'}")
    print(f"  {'─'*55}")

    for i in range(num_pairs):
        start_configs, goal_configs = env.sample_start_goal()
        t0 = time.time()
        paths = planner.plan(env, start_configs, goal_configs)
        elapsed = time.time() - t0
        times.append(elapsed)

        if paths is None:
            print(f"  {i+1:<5} {elapsed:>8.4f}s {'—':>9}  {'—':<20}  FAILED")
        else:
            total_len = sum(
                sum(np.linalg.norm(np.array(p[j+1]) - np.array(p[j]))
                    for j in range(len(p)-1))
                for p in paths
            )
            wps = str([len(p) for p in paths])
            print(f"  {i+1:<5} {elapsed:>8.4f}s {total_len:>9.2f}  {wps:<20}  OK")
            path_lengths.append(total_len)
            solved += 1

    avg_time = np.mean(times)
    max_time = np.max(times)
    min_time = np.min(times)
    avg_len  = np.mean(path_lengths) if path_lengths else float('nan')

    print(f"\n  ✓ SOLVED:   {solved}/{num_pairs}")
    print(f"  ⏱ AvgTime:  {avg_time:.4f}s   MinTime: {min_time:.4f}s   MaxTime: {max_time:.4f}s")
    print(f"  📏 AvgLen:   {avg_len:.2f}")
    print(f"{'─'*60}\n")

    return {"env": env_id, "solved": solved, "total": num_pairs,
            "avg_time": avg_time, "max_time": max_time, "avg_len": avg_len}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",   type=int, default=1)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--all",   action="store_true", help="Run all 4 envs")
    args = parser.parse_args()

    if args.all:
        summary = []
        for e in [1, 2, 3, 4]:
            r = benchmark(e, args.pairs, args.seed)
            summary.append(r)
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║  FULL SUMMARY                                        ║")
        print("╠══════════════════════════════════════════════════════╣")
        for r in summary:
            print(f"║  ENV {r['env']}: {r['solved']}/{r['total']} solved  "
                  f"AvgT={r['avg_time']:.3f}s  MaxT={r['max_time']:.3f}s  "
                  f"AvgLen={r['avg_len']:.1f}")
        print("╚══════════════════════════════════════════════════════╝")
    else:
        benchmark(args.env, args.pairs, args.seed)
