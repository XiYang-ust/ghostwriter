from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from ghostwriter.core import (
    extract_rating,
    extract_statement,
    inject_statement,
    repackage_statement,
)
from ghostwriter.examples import EXAMPLES
from ghostwriter.providers import DEEPSEEK_API_URL, DeepSeekChatModel, OpenAIChatModel
from ghostwriter import inject_cli, repackage_cli


class FakeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, messages, *, temperature, max_tokens):
        self.calls.append(
            {
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.outputs.pop(0)


class CoreTests(unittest.TestCase):
    def test_extract_statement_from_wrapped_json(self):
        value = extract_statement('prefix {"statement": "refined view"} suffix')
        self.assertEqual(value, "refined view")

    def test_extract_rating(self):
        self.assertEqual(extract_rating("Rating: [10]"), 10)
        self.assertEqual(extract_rating("rating: 7"), 7)

    def test_repackaging_stops_at_threshold_and_preserves_feedback_context(self):
        attacker = FakeModel(
            [
                '{"statement": "candidate one"}',
                '{"statement": "candidate two"}',
            ]
        )
        judge = FakeModel(["Rating: [5]", "Rating: [8]"])

        result = repackage_statement(
            "original view",
            attacker,
            judge,
            threshold=8,
            max_rounds=5,
        )

        self.assertEqual(result.statement, "candidate two")
        self.assertEqual(result.score, 8)
        self.assertTrue(result.passed_threshold)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(len(attacker.calls[1]["messages"]), 3)

    def test_repackaging_falls_back_to_best_candidate(self):
        attacker = FakeModel(
            [
                '{"statement": "candidate one"}',
                '{"statement": "candidate two"}',
            ]
        )
        judge = FakeModel(["Rating: [6]", "Rating: [7]"])

        result = repackage_statement(
            "original view",
            attacker,
            judge,
            threshold=8,
            max_rounds=2,
        )

        self.assertEqual(result.statement, "candidate two")
        self.assertEqual(result.score, 7)
        self.assertFalse(result.passed_threshold)

    def test_injection_builds_one_target_message(self):
        target = FakeModel(["target response"])
        response = inject_statement("refined view", "user question", target)

        self.assertEqual(response, "target response")
        prompt = target.calls[0]["messages"][0]["content"]
        self.assertIn("refined view", prompt)
        self.assertIn("user question", prompt)


class ProviderTests(unittest.TestCase):
    def _fake_openai_module(self, request, client):
        class FakeCompletions:
            def create(self, **kwargs):
                request.update(kwargs)
                message = SimpleNamespace(content="provider response")
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        class FakeOpenAI:
            def __init__(self, **kwargs):
                client.update(kwargs)
                self.chat = SimpleNamespace(completions=FakeCompletions())

        return SimpleNamespace(OpenAI=FakeOpenAI)

    def test_openai_adapter_uses_current_token_parameter(self):
        request = {}
        client = {}
        with patch.dict("os.environ", {"TEST_API_KEY": "test-only"}), patch.dict(
            "sys.modules", {"openai": self._fake_openai_module(request, client)}
        ):
            model = OpenAIChatModel("test-model", api_key_env="TEST_API_KEY")
            output = model.complete(
                [{"role": "user", "content": "hello"}],
                temperature=0.0,
                max_tokens=42,
            )

        self.assertEqual(output, "provider response")
        self.assertEqual(request["max_completion_tokens"], 42)
        self.assertNotIn("max_tokens", request)
        self.assertNotIn("base_url", client)

    def test_deepseek_adapter_uses_fixed_official_endpoint(self):
        request = {}
        client = {}
        with patch.dict("os.environ", {"TEST_API_KEY": "test-only"}), patch.dict(
            "sys.modules", {"openai": self._fake_openai_module(request, client)}
        ):
            model = DeepSeekChatModel("test-model", api_key_env="TEST_API_KEY")
            model.complete(
                [{"role": "user", "content": "hello"}],
                temperature=0.0,
                max_tokens=17,
            )

        self.assertEqual(client["base_url"], DEEPSEEK_API_URL)
        self.assertEqual(request["max_tokens"], 17)
        self.assertNotIn("max_completion_tokens", request)


class CliTests(unittest.TestCase):
    def test_phase_1_writes_repackaged_result(self):
        attacker = FakeModel(['{"statement": "selected evidence"}'])
        judge = FakeModel(["Rating: [9]"])

        with TemporaryDirectory() as directory:
            output = Path(directory) / "repackaged.json"
            with patch.object(
                repackage_cli,
                "create_chat_model",
                side_effect=[attacker, judge],
            ):
                repackage_cli.main(
                    [
                        "--statement",
                        "original view",
                        "--output",
                        str(output),
                    ]
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["original_statement"], "original view")
        self.assertEqual(payload["repackaged_statement"], "selected evidence")
        self.assertEqual(payload["repackaging_score"], 9)

    def test_phase_1_accepts_builtin_example(self):
        attacker = FakeModel(['{"statement": "selected evidence"}'])
        judge = FakeModel(["Rating: [10]"])
        stdout = StringIO()

        with patch.object(
            repackage_cli,
            "create_chat_model",
            side_effect=[attacker, judge],
        ):
            with redirect_stdout(stdout):
                repackage_cli.main(["--example", "toyota_oxygen"])
        payload = json.loads(stdout.getvalue())

        self.assertEqual(payload["source_example"], "hvd-g-f-010")
        self.assertEqual(
            payload["original_statement"],
            EXAMPLES["toyota_oxygen"].original_statement,
        )
        self.assertEqual(
            payload["suggested_question"],
            EXAMPLES["toyota_oxygen"].question,
        )

    def test_phase_2_reads_phase_1_result(self):
        target = FakeModel(["target response"])

        with TemporaryDirectory() as directory:
            phase_1 = Path(directory) / "repackaged.json"
            phase_1.write_text(
                json.dumps({"repackaged_statement": "selected evidence"}),
                encoding="utf-8",
            )
            stdout = StringIO()
            with patch.object(inject_cli, "create_chat_model", return_value=target):
                with redirect_stdout(stdout):
                    inject_cli.main(
                        [
                            "--input",
                            str(phase_1),
                            "--question",
                            "user question",
                        ]
                    )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(payload["repackaged_statement"], "selected evidence")
        self.assertEqual(payload["response"], "target response")

    def test_phase_2_can_directly_inject_builtin_example(self):
        target = FakeModel(["target response"])
        stdout = StringIO()

        with patch.object(inject_cli, "create_chat_model", return_value=target):
            with redirect_stdout(stdout):
                inject_cli.main(["--example", "ikea_home_value"])
        payload = json.loads(stdout.getvalue())

        self.assertEqual(payload["source_example"], "hvd-g-f-021")
        self.assertEqual(
            payload["repackaged_statement"],
            EXAMPLES["ikea_home_value"].repackaged_statement,
        )
        self.assertEqual(
            payload["user_question"],
            EXAMPLES["ikea_home_value"].question,
        )


if __name__ == "__main__":
    unittest.main()
