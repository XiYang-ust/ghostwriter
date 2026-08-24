"""Command-line entry point for Phase 1: statement repackaging."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .core import repackage_statement
from .examples import EXAMPLE_NAMES, EXAMPLES
from .providers import create_chat_model


PROVIDERS = ("openai", "anthropic", "deepseek")


def _add_model_arguments(
    parser: argparse.ArgumentParser,
    name: str,
    default_model: str,
) -> None:
    parser.add_argument(
        f"--{name}-provider",
        choices=PROVIDERS,
        default="openai",
    )
    parser.add_argument(f"--{name}-model", default=default_model)
    parser.add_argument(f"--{name}-api-key-env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 of the official Ghostwriter implementation."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--statement", help="Original statement to repackage.")
    source.add_argument(
        "--example",
        choices=EXAMPLE_NAMES,
        help="Use a curated non-demographic HVD-G example.",
    )
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the Phase 1 JSON result to this file instead of stdout.",
    )
    _add_model_arguments(parser, "attacker", "gpt-4o-mini")
    _add_model_arguments(parser, "judge", "gpt-4o")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    example = EXAMPLES[args.example] if args.example else None
    statement = example.original_statement if example else args.statement
    attacker = create_chat_model(
        args.attacker_provider,
        args.attacker_model,
        api_key_env=args.attacker_api_key_env,
    )
    judge = create_chat_model(
        args.judge_provider,
        args.judge_model,
        api_key_env=args.judge_api_key_env,
    )
    result = repackage_statement(
        statement,
        attacker,
        judge,
        threshold=args.threshold,
        max_rounds=args.max_rounds,
    )
    payload = {
        "original_statement": statement,
        "repackaged_statement": result.statement,
        "repackaging_score": result.score,
        "passed_threshold": result.passed_threshold,
        "candidates": [asdict(candidate) for candidate in result.candidates],
    }
    if example:
        payload["source_example"] = example.source_id
        payload["suggested_question"] = example.question
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.output is None:
        print(text, end="")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
