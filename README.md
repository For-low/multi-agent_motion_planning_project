# Multi-Agent Motion Planning Project

A local benchmarking and tuning toolkit for prioritized, collision-aware motion planning of multiple UR5 robots.

## Overview

The project provides a multi-variant BiRRT-based planner and utilities for comparing configurations, benchmarking environments, profiling collision checks, and tuning planner parameters.

### Planner variants

- `straight`: tries direct collision-free paths only
- `birrt`: bidirectional RRT-Connect
- `halton`: BiRRT with Halton quasi-random sampling
- `informed`: Halton sampling plus informed ellipsoid sampling
- `full`: informed BiRRT, priority re-scheduling, and path shortcutting

## Requirements

- Python 3.9 or newer
- NumPy
- The course-provided `motion_planning_env` module

Install NumPy if it is not already available:

```bash
pip install numpy
```

## Configuration

Open `planner.py` and set:

```python
DEBUG_FLAG = False
VARIANT = "full"
TEST_ENVS = [1, 2, 3, 4]
```

Supported variant values are `straight`, `birrt`, `halton`, `informed`, and `full`.

## Usage

Benchmark one environment:

```bash
python benchmark.py --env 1 --pairs 10 --seed 42
```

Benchmark all four environments:

```bash
python benchmark.py --all --pairs 10 --seed 42
```

Compare all variants:

```bash
python compare_variants.py --env 1 --pairs 5 --seed 42
```

Profile collision-check cost:

```bash
python profile_collisions.py --env 1 --pairs 5
```

Search for good BiRRT parameters:

```bash
python tune_params.py --env 1 --pairs 5 --seed 42
```

## Project Structure

- `planner.py`: motion planner and public `get_planner()` entry point
- `benchmark.py`: success, timing, and path-length benchmarks
- `compare_variants.py`: side-by-side planner variant comparison
- `profile_collisions.py`: collision-check profiling
- `tune_params.py`: grid search for BiRRT parameters

## Submission

Before VPL submission:

1. Set `DEBUG_FLAG = False` in `planner.py`.
2. Select the intended `VARIANT`.
3. Configure `TEST_ENVS` if needed.
4. Verify the planner with the benchmark scripts.
