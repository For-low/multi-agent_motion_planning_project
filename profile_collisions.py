"""
profile_collisions.py — Count collision checks (the main time cost)
Usage: python profile_collisions.py --env 1 --pairs 5
"""
import time, argparse
import numpy as np
from motion_planning_env import MotionPlanningEnv
from planner import get_planner, VARIANT

class ProfilingEnv:
    """Wraps MotionPlanningEnv to count every collision check call."""
    def __init__(self, env_id):
        self._env = MotionPlanningEnv(env_id)
        self.checks = 0
        self.num_robots = self._env.num_robots

    def check_collision_single_robot(self, ri, q):
        self.checks += 1
        return self._env.check_collision_single_robot(ri, q)

    def sample_start_goal(self):
        return self._env.sample_start_goal()

def profile(env_id, num_pairs=5, seed=42):
    np.random.seed(seed)
    planner = get_planner()
    env = ProfilingEnv(env_id)

    print(f"\n  Profiling ENV {env_id}  VARIANT={VARIANT}")
    print(f"  {'Pair':<5} {'Time':>8}  {'ColChks':>9}  {'Chks/s':>9}  {'PathLen':>9}  Status")
    print(f"  {'─'*60}")

    for i in range(num_pairs):
        env.checks = 0
        start_configs, goal_configs = env.sample_start_goal()
        t0 = time.time()
        paths = planner.plan(env, start_configs, goal_configs)
        elapsed = time.time() - t0
        chk_per_s = env.checks / elapsed if elapsed > 0 else 0

        if paths:
            total_len = sum(
                sum(np.linalg.norm(np.array(p[j+1]) - np.array(p[j]))
                    for j in range(len(p)-1))
                for p in paths
            )
            print(f"  {i+1:<5} {elapsed:>8.4f}s  {env.checks:>9d}  {chk_per_s:>9.0f}  {total_len:>9.2f}  OK")
        else:
            print(f"  {i+1:<5} {elapsed:>8.4f}s  {env.checks:>9d}  {chk_per_s:>9.0f}  {'—':>9}  FAILED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",   type=int, default=1)
    parser.add_argument("--pairs", type=int, default=5)
    args = parser.parse_args()
    profile(args.env, args.pairs)
