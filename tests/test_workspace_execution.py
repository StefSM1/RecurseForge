import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from harness.workspace_execution import (
    WorkspaceExecutionService,
    WorkspaceExecutionSettings,
)
from harness.workspace_service import WorkspaceError, WorkspaceService, WorkspaceSettings


class WorkspaceExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = WorkspaceService(WorkspaceSettings(
            root=Path(self.temp.name) / ".recurseforge",
        ))
        self.executor = WorkspaceExecutionService(
            self.service,
            WorkspaceExecutionSettings(python_executable=Path(sys.executable), timeout_s=1),
        )

    def tearDown(self):
        self.temp.cleanup()

    def _write_manifest(self, entrypoints=None, tests=None):
        manifest = {
            "schema_version": 1,
            "entrypoints": entrypoints or [],
            "tests": tests or [],
        }
        self.service.create_file("recurseforge.json", json.dumps(manifest))

    def test_load_manifest_accepts_python_scripts_and_modules(self):
        self.service.create_file("main.py", "print('hello')")
        self.service.create_file("tests/test_main.py", "import unittest\n")
        self._write_manifest(
            entrypoints=[{"id": "main", "kind": "script", "target": "main.py", "args": []}],
            tests=[{
                "id": "unit",
                "kind": "module",
                "target": "unittest",
                "args": ["discover", "-s", "tests"],
            }],
        )

        manifest = self.executor.load_manifest()

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["entrypoints"][0]["target"], "main.py")
        self.assertEqual(manifest["tests"][0]["kind"], "module")

    def test_rejects_shell_targets_command_strings_and_duplicate_ids(self):
        self.service.create_file("main.py", "print('hello')")
        self.service.create_file("tests/test_main.py", "import unittest\n")
        self._write_manifest(
            entrypoints=[{"id": "run", "kind": "shell", "target": "python main.py", "args": []}],
            tests=[{"id": "unit", "kind": "module", "target": "unittest", "args": []}],
        )
        with self.assertRaises(WorkspaceError) as unsupported:
            self.executor.load_manifest()
        self.assertEqual(unsupported.exception.code, "unsupported_target")

        self.service.archive_reset(confirmed=True, run_id="test")
        self.service.create_file("main.py", "print('hello')")
        self.service.create_file("tests/test_main.py", "import unittest\n")
        self._write_manifest(
            entrypoints=[{"id": "run", "kind": "module", "target": "unittest discover", "args": []}],
            tests=[{"id": "unit", "kind": "module", "target": "unittest", "args": []}],
        )
        with self.assertRaises(WorkspaceError) as command_string:
            self.executor.load_manifest()
        self.assertEqual(command_string.exception.code, "invalid_manifest")

        self.service.archive_reset(confirmed=True, run_id="test")
        self.service.create_file("main.py", "print('hello')")
        self.service.create_file("tests/test_main.py", "import unittest\n")
        self._write_manifest(
            entrypoints=[{"id": "same", "kind": "script", "target": "main.py", "args": []}],
            tests=[{"id": "same", "kind": "module", "target": "unittest", "args": []}],
        )
        with self.assertRaises(WorkspaceError) as duplicate:
            self.executor.load_manifest()
        self.assertEqual(duplicate.exception.code, "invalid_manifest")

    def test_python_workspaces_require_tests_but_docs_only_do_not(self):
        self.service.create_file("main.py", "print('hello')")
        self._write_manifest(
            entrypoints=[{"id": "main", "kind": "script", "target": "main.py", "args": []}],
            tests=[],
        )
        with self.assertRaises(WorkspaceError) as missing_tests:
            self.executor.load_manifest()
        self.assertEqual(missing_tests.exception.code, "tests_required")

        self.service.archive_reset(confirmed=True, run_id="test")
        self.service.create_file("README.md", "# Notes\n")
        self._write_manifest(entrypoints=[], tests=[])
        self.assertEqual(self.executor.load_manifest()["tests"], [])

    def test_success_failure_and_timeout_results(self):
        self.service.create_file("ok.py", "print('ok')")
        self.service.create_file("fail.py", "raise SystemExit(3)")
        self.service.create_file("slow.py", "import time\ntime.sleep(5)")
        self.service.create_file("tests/test_smoke.py", "import unittest\n")
        self._write_manifest(
            entrypoints=[
                {"id": "ok", "kind": "script", "target": "ok.py", "args": []},
                {"id": "fail", "kind": "script", "target": "fail.py", "args": []},
                {"id": "slow", "kind": "script", "target": "slow.py", "args": []},
            ],
            tests=[{"id": "unit", "kind": "module", "target": "unittest", "args": ["discover", "-s", "tests"]}],
        )

        ok = self.executor.run_python_target("ok")
        failed = self.executor.run_python_target("fail")
        started = time.perf_counter()
        timed_out = self.executor.run_python_target("slow")

        self.assertEqual(ok["status"], "success")
        self.assertEqual(ok["stdout"], "ok\n")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["exit_code"], 3)
        self.assertEqual(timed_out["status"], "timeout")
        self.assertLess(time.perf_counter() - started, 3)

    def test_snapshot_mutations_do_not_change_authoritative_workspace(self):
        self.service.create_file(
            "main.py",
            "from pathlib import Path\nPath('generated.txt').write_text('temp')\nPath('main.py').write_text('mutated')\nprint('done')\n",
        )
        self.service.create_file("tests/test_smoke.py", "import unittest\n")
        self._write_manifest(
            entrypoints=[{"id": "main", "kind": "script", "target": "main.py", "args": []}],
            tests=[{"id": "unit", "kind": "module", "target": "unittest", "args": ["discover", "-s", "tests"]}],
        )
        before_revision = self.service.workspace_revision

        result = self.executor.run_python_target("main")

        self.assertEqual(result["snapshot_revision"], before_revision)
        self.assertEqual(result["status"], "success")
        self.assertIn("done", result["stdout"])
        self.assertIn("write_text('mutated')", self.service.read_file("main.py")["content"])
        with self.assertRaises(WorkspaceError) as generated:
            self.service.read_file("generated.txt")
        self.assertEqual(generated.exception.code, "not_found")
        self.assertEqual(self.service.workspace_revision, before_revision)


if __name__ == "__main__":
    unittest.main()
