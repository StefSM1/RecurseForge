import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from harness import dashboard_server
from harness.workspace_service import WorkspaceService, WorkspaceSettings


class WorkspaceApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = dashboard_server._workspace_service
        dashboard_server._workspace_service = WorkspaceService(
            WorkspaceSettings(root=Path(self.temp.name) / ".recurseforge")
        )

    def tearDown(self):
        dashboard_server._workspace_service = self.previous
        self.temp.cleanup()

    async def test_file_lifecycle_through_api_handlers(self):
        created = await dashboard_server.workspace_create_file({
            "path": "src/main.py", "content": "print('hello')",
        })
        tree = await dashboard_server.workspace_tree()
        self.assertEqual(tree["files"][0]["path"], "src/main.py")
        read = await dashboard_server.workspace_read_file("src/main.py")
        self.assertEqual(read["content"], "print('hello')")
        saved = await dashboard_server.workspace_save_file({
            "path": "src/main.py",
            "content": "print('updated')",
            "expected_revision": created["revision"],
        })
        self.assertNotEqual(saved["revision"], created["revision"])
        deleted = await dashboard_server.workspace_delete_file("src/main.py")
        trash = await dashboard_server.workspace_trash()
        self.assertEqual(trash["items"][0]["operation_id"], deleted["operation_id"])
        await dashboard_server.workspace_restore_trash(deleted["operation_id"])
        self.assertEqual((await dashboard_server.workspace_read_file("src/main.py"))["content"], "print('updated')")

    async def test_api_error_shape_and_destructive_confirmation(self):
        with self.assertRaises(HTTPException) as traversal:
            await dashboard_server.workspace_read_file("../secret.txt")
        self.assertEqual(traversal.exception.detail["code"], "invalid_path")

        with self.assertRaises(HTTPException) as confirmation:
            await dashboard_server.workspace_archive_reset({"confirm": False})
        self.assertEqual(confirmation.exception.detail["code"], "confirmation_required")
        archived = await dashboard_server.workspace_archive_reset({
            "confirm": True, "run_id": "api-test",
        })
        history = await dashboard_server.workspace_history()
        self.assertEqual(history["archives"][0]["archive_id"], archived["archive_id"])

    async def test_manifest_and_python_target_api_handlers(self):
        created = await dashboard_server.workspace_create_file({
            "path": "main.py",
            "content": "print('api ok')",
        })
        await dashboard_server.workspace_create_file({
            "path": "tests/test_smoke.py",
            "content": "import unittest\n",
        })
        await dashboard_server.workspace_create_file({
            "path": "recurseforge.json",
            "content": (
                '{"schema_version": 1, '
                '"entrypoints": [{"id": "main", "kind": "script", "target": "main.py", "args": []}], '
                '"tests": [{"id": "unit", "kind": "module", "target": "unittest", "args": ["discover", "-s", "tests"]}]}'
            ),
        })

        manifest = await dashboard_server.workspace_manifest()
        self.assertEqual(manifest["entrypoints"][0]["id"], "main")
        result = await dashboard_server.workspace_run_python_target({"target_id": "main"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stdout"], "api ok\n")
        self.assertEqual((await dashboard_server.workspace_read_file("main.py"))["revision"], created["revision"])


if __name__ == "__main__":
    unittest.main()
