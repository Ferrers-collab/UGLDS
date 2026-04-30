from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from datetime import datetime

from .core import load_routing_context
from .problem1 import Problem1Config, solve_problem1, write_problem1_outputs
from .problem2 import Problem2Config, solve_problem2, write_problem2_outputs
from .problem3 import Problem3Config, solve_problem3, write_problem3_outputs
from .problem3_benchmark import run_problem3_benchmarks


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def p1_progress(generation: int, total: int, best_cost: float) -> None:
    log(f"问题1进度 {generation}/{total}，当前最优成本={best_cost:.2f}")


def p2_progress(generation: int, total: int, feasible_count: int, best_cost: float, best_carbon: float) -> None:
    cost_text = 'inf' if best_cost == float('inf') else f'{best_cost:.2f}'
    carbon_text = 'inf' if best_carbon == float('inf') else f'{best_carbon:.2f}'
    log(f"问题2进度 {generation}/{total}，可行解={feasible_count}，当前最优成本={cost_text}，最优碳排={carbon_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="城市绿色物流配送调度求解器")
    parser.add_argument("--problem", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--output-dir", default="model/results")
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--p1-population", type=int, default=60)
    parser.add_argument("--p1-generations", type=int, default=120)
    parser.add_argument("--p2-population", type=int, default=60)
    parser.add_argument("--p2-generations", type=int, default=120)
    parser.add_argument("--p3-iterations", type=int, default=8)
    parser.add_argument("--p3-time-limit", type=float, default=8.0, help="问题3 LNS 改善阶段的秒级时间上限")
    parser.add_argument("--p3-benchmark", action="store_true", help="运行问题3多场景实时性评测")
    parser.add_argument("--p3-benchmark-runs", type=int, default=3, help="每个问题3场景重复次数")
    parser.add_argument("--quiet", action="store_true", help="关闭运行过程中的进度提示")
    parser.add_argument(
        "--reuse-p1",
        action="store_true",
        help="兼容参数：若存在 output-dir/problem1/baseline_solution.pkl，则优先复用，避免重复求解问题1",
    )
    parser.add_argument(
        "--recompute-p1",
        action="store_true",
        help="强制重新求解问题1，不使用已有 baseline_solution.pkl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.quiet:
        log("开始加载数据与模型上下文...")
    context = load_routing_context()
    if not args.quiet:
        log(f"数据加载完成：客户任务数={len(context.jobs)}，车辆类型数={len(context.vehicle_types)}")
    p1_cache_path = output_dir / "problem1" / "baseline_solution.pkl"

    baseline_solution = None

    if args.problem in {"1", "all", "2", "3"}:
        if (not args.recompute_p1) and p1_cache_path.exists():
            if not args.quiet:
                log(f"发现问题1缓存，直接复用：{p1_cache_path}")
            baseline_solution = pickle.loads(p1_cache_path.read_bytes())
        else:
            p1_config = Problem1Config(
                seed=args.seed,
                population_size=args.p1_population,
                generations=args.p1_generations,
                progress_callback=None if args.quiet else p1_progress,
            )
            if not args.quiet:
                log(f"开始求解问题1：population={args.p1_population}, generations={args.p1_generations}")
            baseline_solution = solve_problem1(context, p1_config)
            if not args.quiet:
                log(f"问题1求解完成：总成本={baseline_solution.total_cost:.2f}，路线数={len(baseline_solution.routes)}")
            p1_cache_path.parent.mkdir(parents=True, exist_ok=True)
            p1_cache_path.write_bytes(pickle.dumps(baseline_solution))
        if args.problem in {"1", "all"}:
            write_problem1_outputs(baseline_solution, output_dir / "problem1")
            if not args.quiet:
                log(f"问题1结果已写入：{output_dir / 'problem1'}")

    if args.problem in {"2", "all"}:
        p2_config = Problem2Config(
            seed=args.seed + 1,
            population_size=args.p2_population,
            generations=args.p2_generations,
            progress_callback=None if args.quiet else p2_progress,
        )
        if not args.quiet:
            log(f"开始求解问题2：population={args.p2_population}, generations={args.p2_generations}")
        result2 = solve_problem2(context, baseline_solution, p2_config)
        if not args.quiet:
            selected = result2.get("selected_solution")
            if selected is not None:
                log(f"问题2求解完成：总成本={selected.total_cost:.2f}，碳排={selected.total_carbon:.2f}")
        write_problem2_outputs(result2, output_dir / "problem2")
        if not args.quiet:
            log(f"问题2结果已写入：{output_dir / 'problem2'}")

    if args.problem in {"3", "all"}:
        p3_config = Problem3Config(seed=args.seed + 2, search_iterations=args.p3_iterations, time_limit_seconds=args.p3_time_limit)
        if not args.quiet:
            log("开始求解问题3：快速动态修复/重调度")
        result3 = solve_problem3(context, baseline_solution, p3_config)
        if not args.quiet:
            sol = result3.get("solution")
            if sol is not None:
                log(f"问题3求解完成：估计全天累计成本={sol.metadata.get('full_day_cost_est', sol.total_cost):.2f}，总耗时={sol.metadata.get('solve_seconds', 0):.4f}s")
        write_problem3_outputs(result3, output_dir / "problem3")
        if not args.quiet:
            log(f"问题3结果已写入：{output_dir / 'problem3'}")
        if args.p3_benchmark:
            if not args.quiet:
                log("开始运行问题3 benchmark...")
            run_problem3_benchmarks(
                output_dir=output_dir / "problem3",
                runs_per_scenario=max(1, int(args.p3_benchmark_runs)),
                seed=args.seed,
                context=context,
                baseline_solution=baseline_solution,
            )
            if not args.quiet:
                log("问题3 benchmark 完成")

    if not args.quiet:
        log("全部任务完成。")


if __name__ == "__main__":
    main()
