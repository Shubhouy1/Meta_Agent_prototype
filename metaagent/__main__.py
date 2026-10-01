"""Command-line entry point.

    python -m metaagent plan  "Build a PDF QA system"
    python -m metaagent build "Build a calculator that can do basic math" --budget free
    python -m metaagent build "..." --no-cache

Replaces the old planner_test.py / tempCodeRunnerFile.py scripts.
"""

import argparse
import json
import sys

from metaagent.builds.service import BuildService
from metaagent.core.config import get_settings
from metaagent.core.logging import configure_logging
from metaagent.schemas import BuildEvent, Constraints


def main(argv=None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="metaagent")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_cmd = sub.add_parser("plan", help="run Stage 1 only")
    plan_cmd.add_argument("request")
    plan_cmd.add_argument("--model", choices=settings.available_models)

    build_cmd = sub.add_parser("build", help="run the full pipeline")
    build_cmd.add_argument("request")
    build_cmd.add_argument("--model", choices=settings.available_models)
    build_cmd.add_argument("--budget", choices=["free", "paid"], default="free")
    build_cmd.add_argument("--privacy", choices=["strict", "moderate", "none"], default="moderate")
    build_cmd.add_argument("--performance", choices=["fast", "balanced"], default="balanced")
    build_cmd.add_argument("--no-cache", action="store_true")

    args = parser.parse_args(argv)
    configure_logging()

    if args.command == "plan":
        from metaagent.ai.planner import generate_plan

        print(generate_plan(args.request, model=args.model).model_dump_json(indent=2))
        return 0

    def show(event: BuildEvent) -> None:
        print(f"[{event.stage}] {event.event} {event.message}".rstrip(), file=sys.stderr)

    result = BuildService(settings).start_build(
        args.request,
        Constraints(budget=args.budget, privacy=args.privacy, performance=args.performance),
        args.model, on_event=show, use_cache=not args.no_cache,
    )
    summary = {
        "build_id": result.build_id,
        "status": result.status,
        "failed_stage": result.failed_stage,
        "error": result.error.message if result.error else None,
        "agent_type": result.plan.plan.agent_type if result.plan else None,
        "method": result.generation.method if result.generation else None,
        "corrections": result.generation.corrections if result.generation else None,
        "checks": [f"{'PASS' if c.passed else 'FAIL'} {c.name}: {c.message}" for c in result.test.checks]
        if result.test else [],
        "deployed_file": result.deployment.deployed_file if result.deployed else None,
        "duration_s": round(result.duration_s, 1),
    }
    print(json.dumps(summary, indent=2))
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
