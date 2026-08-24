"""Command-line entry point for Phase 2: statement injection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .core import inject_statement
from .examples import EXAMPLE_NAMES, EXAMPLES
from .providers import create_chat_model


PROVIDERS = ("openai", "anthropic", "deepseek")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 of the official Ghostwriter implementation."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="Phase 1 JSON file produced by ghostwriter-repackage.",
    )
    source.add_argument(
        "--example",
        choices=EXAMPLE_NAMES,
        help="Directly inject a curated repackaged HVD-G example.",
    )
    parser.add_argument(
        "--question",
        help="User question; built-in examples provide a default.",
    )
    parser.add_argument(
        "--target-provider",
        choices=PROVIDERS,
        default="openai",
    )
    parser.add_argument("--target-model", default="gpt-4o")
    parser.add_argument("--target-api-key-env")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the Phase 2 JSON result to this file instead of stdout.",
    )
    return parser


def _read_phase_1_result(path: Path) -> tuple[str, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Phase 1 JSON from {path}: {exc}") from exc

    statement = payload.get("repackaged_statement") if isinstance(payload, dict) else None
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError(
            f"Phase 1 JSON at {path} does not contain a non-empty repackaged_statement"
        )
    suggested_question = payload.get("suggested_question")
    if not isinstance(suggested_question, str) or not suggested_question.strip():
        suggested_question = None
    return statement.strip(), suggested_question


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    example = EXAMPLES[args.example] if args.example else None
    source_example = None

    if example:
        repackaged_statement = example.repackaged_statement
        question = args.question or example.question
        source_example = example.source_id
    else:
        try:
            repackaged_statement, suggested_question = _read_phase_1_result(args.input)
        except ValueError as exc:
            parser.error(str(exc))
        question = args.question or suggested_question
        if not question:
            parser.error("--question is required when the Phase 1 file has no suggested_question")

    target = create_chat_model(
        args.target_provider,
        args.target_model,
        api_key_env=args.target_api_key_env,
    )
    response = inject_statement(repackaged_statement, question, target)
    payload = {
        "repackaged_statement": repackaged_statement,
        "user_question": question,
        "response": response,
    }
    if source_example:
        payload["source_example"] = source_example
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.output is None:
        print(text, end="")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
