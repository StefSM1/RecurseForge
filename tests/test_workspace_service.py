import os
import tempfile
import threading
import unittest
from pathlib import Path

from harness import event_bus
from harness.workspace_service import (
    WorkspaceConflict,
    WorkspaceError,
    WorkspaceService,
    WorkspaceSettings,
)


class WorkspaceServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        event_bus._bus = event_bus.EventBus()
        self.service = WorkspaceService(WorkspaceSettings(
            root=Path(self.temp.name) / ".recurseforge",
            max_file_bytes=32,
            max_workspace_bytes=64,
        ))

    def tearDown(self):
        self.temp.cleanup()

    def test_create_read_save_and_persistent_revision(self):
        created = self.service.create_file("src/main.py", "print('a')")
        read = self.service.read_file("src/main.py")
        self.assertEqual(read["content"], "print('a')")
        saved = self.service.save_file("src/main.py", "print('b')", created["revision"])
        self.assertNotEqual(saved["revision"], created["revision"])
        restarted = WorkspaceService(self.service.settings)
        self.assertEqual(restarted.tree()["workspace_revision"], 2)
        events = event_bus.get_event_bus().drain()
        self.assertEqual([event.event_type for event in events], [
            "workspace_changed", "workspace_changed",
        ])
        self.assertNotIn("content", events[-1].payload)

    def test_rejects_traversal_absolute_binary_and_size_limits(self):
        for path in ("../escape.py", "/absolute.py", "C:/absolute.py"):
            with self.subTest(path=path), self.assertRaises(WorkspaceError):
                self.service.create_file(path, "x")
        with self.assertRaises(WorkspaceError) as size:
            self.service.create_file("large.txt", "x" * 33)
        self.assertEqual(size.exception.code, "file_too_large")

        binary = self.service.active / "binary.dat"
        binary.write_bytes(b"a\x00b")
        with self.assertRaises(WorkspaceError) as binary_error:
            self.service.read_file("binary.dat")
        self.assertEqual(binary_error.exception.code, "binary_rejected")

    def test_rejects_symlinks_when_supported(self):
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        link = self.service.active / "link.txt"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable for this Windows account")
        with self.assertRaises(WorkspaceError) as error:
            self.service.read_file("link.txt")
        self.assertEqual(error.exception.code, "symlink_rejected")

    def test_stale_revision_exact_edit_and_lock_ownership(self):
        created = self.service.create_file("main.py", "value = 1")
        with self.assertRaises(WorkspaceConflict) as stale:
            self.service.save_file("main.py", "value = 2", "stale")
        self.assertEqual(stale.exception.code, "stale_revision")
        with self.assertRaises(WorkspaceConflict) as ambiguous:
            self.service.edit_file("main.py", created["revision"], [{
                "old_text": "missing", "new_text": "x", "expected_occurrences": 1,
            }])
        self.assertEqual(ambiguous.exception.code, "ambiguous_edit")

        self.service.acquire_lock("main.py", "worker-1")
        with self.assertRaises(WorkspaceConflict):
            self.service.save_file("main.py", "value = 2", created["revision"], owner_id="user")
        saved = self.service.save_file("main.py", "value = 2", created["revision"], owner_id="worker-1")
        self.assertTrue(saved["revision"])
        self.service.release_lock("main.py", "worker-1")

    def test_trash_restore_refuses_overwrite_and_permanent_delete_confirms(self):
        self.service.create_file("notes.txt", "old")
        deleted = self.service.delete_file("notes.txt")
        self.assertEqual(len(self.service.list_trash()["items"]), 1)
        self.service.create_file("notes.txt", "new")
        with self.assertRaises(WorkspaceConflict):
            self.service.restore_trash(deleted["operation_id"])
        with self.assertRaises(WorkspaceError) as confirmation:
            self.service.permanently_delete_trash(deleted["operation_id"], confirmed=False)
        self.assertEqual(confirmation.exception.code, "confirmation_required")
        self.service.permanently_delete_trash(deleted["operation_id"], confirmed=True)

    def test_archive_reset_and_restore_archive_current_first(self):
        self.service.create_file("first.txt", "first")
        with self.assertRaises(WorkspaceError):
            self.service.archive_reset(confirmed=False)
        archived = self.service.archive_reset(confirmed=True, run_id="test")
        self.assertEqual(self.service.tree()["files"], [])
        self.service.create_file("second.txt", "second")
        restored = self.service.restore_history(archived["archive_id"], confirmed=True)
        self.assertEqual(self.service.read_file("first.txt")["content"], "first")
        self.assertFalse((self.service.active / "second.txt").exists())
        self.assertTrue(restored["current_archive_id"])

    def test_workspace_limit_and_concurrent_creates(self):
        self.service.create_file("a.txt", "a" * 32)
        self.service.create_file("b.txt", "b" * 32)
        with self.assertRaises(WorkspaceError) as limit:
            self.service.create_file("c.txt", "c")
        self.assertEqual(limit.exception.code, "workspace_too_large")

        other = WorkspaceService(WorkspaceSettings(root=Path(self.temp.name) / "other"))
        errors = []
        def create(index):
            try:
                other.create_file(f"file-{index}.txt", str(index))
            except Exception as exc:  # pragma: no cover - assertion reports details
                errors.append(exc)
        threads = [threading.Thread(target=create, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(other.tree()["files"]), 8)


if __name__ == "__main__":
    unittest.main()
