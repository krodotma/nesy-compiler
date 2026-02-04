#!/usr/bin/env python3
"""
Unit tests for membrane_manager.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from membrane_manager import (
    MembraneEntry,
    MembraneStatus,
    MembraneManager,
    ManifestManager,
    GitOps,
    GitError,
    EntryStatus,
    MembraneType,
    format_status_table,
)


class TestMembraneEntry(unittest.TestCase):
    """Test MembraneEntry dataclass."""

    def test_to_dict_submodule(self):
        """Test converting submodule entry to dict."""
        entry = MembraneEntry(
            name="graphiti",
            type="submodule",
            remote="https://github.com/getzep/graphiti",
            pinned="v0.3.2",
            adapter="tools/graphiti_bridge.py",
        )
        d = entry.to_dict()
        self.assertEqual(d["type"], "submodule")
        self.assertEqual(d["remote"], "https://github.com/getzep/graphiti")
        self.assertEqual(d["pinned"], "v0.3.2")
        self.assertEqual(d["adapter"], "tools/graphiti_bridge.py")
        self.assertNotIn("prefix", d)

    def test_to_dict_subtree(self):
        """Test converting subtree entry to dict."""
        entry = MembraneEntry(
            name="mem0",
            type="subtree",
            remote="https://github.com/mem0ai/mem0",
            pinned="v0.1.8",
            prefix="membrane/mem0-fork",
            adapter="tools/mem0_adapter.py",
        )
        d = entry.to_dict()
        self.assertEqual(d["type"], "subtree")
        self.assertEqual(d["prefix"], "membrane/mem0-fork")

    def test_from_dict_basic(self):
        """Test creating entry from dict."""
        data = {
            "type": "submodule",
            "remote": "https://github.com/test/repo",
            "pinned": "v1.0.0",
            "adapter": "tools/adapter.py",
        }
        entry = MembraneEntry.from_dict("test", data)
        self.assertEqual(entry.name, "test")
        self.assertEqual(entry.type, "submodule")
        self.assertEqual(entry.pinned, "v1.0.0")
        self.assertEqual(entry.adapter, "tools/adapter.py")

    def test_from_dict_planned_submodule(self):
        """Test normalizing planned_submodule type."""
        data = {
            "type": "planned_submodule",
            "remote": "https://github.com/test/repo",
            "pinned": None,
            "recommended_version": "latest",
        }
        entry = MembraneEntry.from_dict("test", data)
        self.assertEqual(entry.type, "submodule")
        self.assertEqual(entry.pinned, "latest")

    def test_from_dict_planned_subtree(self):
        """Test normalizing planned_subtree type."""
        data = {
            "type": "planned_subtree",
            "remote": "https://github.com/test/repo",
            "pinned": None,
            "prefix": "membrane/test-fork",
        }
        entry = MembraneEntry.from_dict("test", data)
        self.assertEqual(entry.type, "subtree")
        self.assertEqual(entry.prefix, "membrane/test-fork")

    def test_from_dict_missing_remote(self):
        """Test handling missing remote."""
        data = {"type": "submodule", "pinned": "v1.0.0"}
        entry = MembraneEntry.from_dict("test", data)
        self.assertEqual(entry.remote, "")


class TestManifestManager(unittest.TestCase):
    """Test ManifestManager class."""

    def setUp(self):
        """Set up temp directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.manifest_path = Path(self.temp_dir) / ".clade-manifest.json"

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_default_manifest(self):
        """Test creating default manifest."""
        manager = ManifestManager(Path(self.temp_dir))
        manifest = manager.load()

        self.assertEqual(manifest["schema_version"], 1)
        self.assertIn("phi", manifest)
        self.assertEqual(manifest["trunk"], "main")
        self.assertIn("membrane", manifest)
        self.assertIn("clades", manifest)

    def test_save_and_load(self):
        """Test saving and loading manifest."""
        manager = ManifestManager(Path(self.temp_dir))

        manifest = manager.load()
        manifest["membrane"]["test"] = {
            "type": "submodule",
            "remote": "https://github.com/test/repo",
            "pinned": "v1.0.0",
        }
        manager.save(manifest)

        # Load again and verify
        loaded = manager.load()
        self.assertIn("test", loaded["membrane"])
        self.assertEqual(loaded["membrane"]["test"]["pinned"], "v1.0.0")

    def test_get_membrane_entries(self):
        """Test getting membrane entries as objects."""
        manager = ManifestManager(Path(self.temp_dir))
        manifest = {
            "membrane": {
                "tool1": {
                    "type": "submodule",
                    "remote": "https://github.com/a/b",
                    "pinned": "v1.0",
                },
                "tool2": {
                    "type": "subtree",
                    "remote": "https://github.com/c/d",
                    "pinned": "v2.0",
                    "prefix": "membrane/tool2-fork",
                },
            }
        }
        manager.save(manifest)

        entries = manager.get_membrane_entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries["tool1"].type, "submodule")
        self.assertEqual(entries["tool2"].type, "subtree")

    def test_set_membrane_entry(self):
        """Test adding/updating a membrane entry."""
        manager = ManifestManager(Path(self.temp_dir))

        entry = MembraneEntry(
            name="new-tool",
            type="submodule",
            remote="https://github.com/new/tool",
            pinned="v1.0.0",
            adapter="tools/new_adapter.py",
        )
        manager.set_membrane_entry(entry)

        entries = manager.get_membrane_entries()
        self.assertIn("new-tool", entries)
        self.assertEqual(entries["new-tool"].pinned, "v1.0.0")

    def test_remove_membrane_entry(self):
        """Test removing a membrane entry."""
        manager = ManifestManager(Path(self.temp_dir))

        # Add entry first
        entry = MembraneEntry(
            name="to-remove",
            type="submodule",
            remote="https://github.com/test/repo",
            pinned="v1.0.0",
        )
        manager.set_membrane_entry(entry)

        # Remove it
        result = manager.remove_membrane_entry("to-remove")
        self.assertTrue(result)

        # Verify removal
        entries = manager.get_membrane_entries()
        self.assertNotIn("to-remove", entries)

    def test_remove_nonexistent_entry(self):
        """Test removing entry that doesn't exist."""
        manager = ManifestManager(Path(self.temp_dir))
        result = manager.remove_membrane_entry("nonexistent")
        self.assertFalse(result)


class TestFormatStatusTable(unittest.TestCase):
    """Test table formatting."""

    def test_empty_list(self):
        """Test formatting empty status list."""
        result = format_status_table([])
        self.assertEqual(result, "No membrane entries found.")

    def test_single_entry_ok(self):
        """Test formatting single OK entry."""
        entry = MembraneEntry(
            name="test",
            type="submodule",
            remote="https://github.com/test/repo",
            pinned="v1.0.0",
        )
        status = MembraneStatus(entry=entry, status=EntryStatus.OK)
        result = format_status_table([status])

        self.assertIn("test", result)
        self.assertIn("submodule", result)
        self.assertIn("v1.0.0", result)
        self.assertIn("[OK]", result)

    def test_entry_with_none_pinned(self):
        """Test formatting entry with None pinned version."""
        entry = MembraneEntry(
            name="test",
            type="submodule",
            remote="https://github.com/test/repo",
            pinned=None,
        )
        status = MembraneStatus(entry=entry, status=EntryStatus.MISSING, message="Not installed")
        result = format_status_table([status])

        self.assertIn("(not pinned)", result)
        self.assertIn("[MISSING]", result)

    def test_multiple_statuses(self):
        """Test formatting multiple entries."""
        entries = [
            MembraneEntry("a", "submodule", "url1", "v1"),
            MembraneEntry("b", "subtree", "url2", "v2", prefix="p"),
            MembraneEntry("c", "submodule", "url3", None),
        ]
        statuses = [
            MembraneStatus(entries[0], EntryStatus.OK),
            MembraneStatus(entries[1], EntryStatus.DIRTY, "Has changes"),
            MembraneStatus(entries[2], EntryStatus.ERROR, "Failed"),
        ]
        result = format_status_table(statuses)

        self.assertIn("[OK]", result)
        self.assertIn("[DIRTY]", result)
        self.assertIn("[ERROR]", result)


class TestGitOpsBasic(unittest.TestCase):
    """Test basic GitOps functionality."""

    def test_git_not_found(self):
        """Test error when git is not found."""
        with patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                GitOps(Path("/tmp"))
            self.assertIn("git executable not found", str(ctx.exception))


class TestMembraneManagerInit(unittest.TestCase):
    """Test MembraneManager initialization."""

    def setUp(self):
        """Set up temp git repo."""
        self.temp_dir = tempfile.mkdtemp()
        subprocess.run(
            ["git", "init"],
            cwd=self.temp_dir,
            capture_output=True,
        )

    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_find_repo_root(self):
        """Test finding repo root."""
        manager = MembraneManager(self.temp_dir)
        self.assertEqual(manager.repo_root, Path(self.temp_dir).resolve())

    def test_invalid_repo(self):
        """Test error for non-git directory."""
        with tempfile.TemporaryDirectory() as non_git:
            with self.assertRaises(ValueError) as ctx:
                MembraneManager(non_git)
            self.assertIn("Not a git repository", str(ctx.exception))


class TestMembraneManagerOperations(unittest.TestCase):
    """Test MembraneManager operations with mocked git."""

    def setUp(self):
        """Set up temp directory with git repo."""
        self.temp_dir = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.temp_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.temp_dir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=self.temp_dir,
            capture_output=True,
        )
        # Create initial commit
        (Path(self.temp_dir) / "README.md").write_text("test")
        subprocess.run(["git", "add", "."], cwd=self.temp_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=self.temp_dir,
            capture_output=True,
        )

    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_list_entries_empty(self):
        """Test listing with no entries."""
        manager = MembraneManager(self.temp_dir)
        statuses = manager.list_entries()
        self.assertEqual(len(statuses), 0)

    def test_verify_adapters_empty(self):
        """Test verifying adapters with no entries."""
        manager = MembraneManager(self.temp_dir)
        results = manager.verify_adapters()
        self.assertEqual(len(results), 0)

    def test_check_updates_empty(self):
        """Test checking updates with no entries."""
        manager = MembraneManager(self.temp_dir)
        results = manager.check_updates()
        self.assertEqual(len(results), 0)

    def test_sync_all_empty(self):
        """Test syncing with no entries."""
        manager = MembraneManager(self.temp_dir)
        results = manager.sync_all()
        self.assertEqual(len(results), 0)

    def test_add_submodule_already_exists(self):
        """Test adding duplicate submodule."""
        manager = MembraneManager(self.temp_dir)

        # Manually add entry to manifest
        entry = MembraneEntry(
            name="existing",
            type="submodule",
            remote="https://github.com/test/repo",
            pinned="v1.0.0",
        )
        manager.manifest.set_membrane_entry(entry)

        # Try to add again
        with self.assertRaises(ValueError) as ctx:
            manager.add_submodule("existing", "https://github.com/other/repo")
        self.assertIn("already exists", str(ctx.exception))

    def test_update_nonexistent(self):
        """Test updating nonexistent entry."""
        manager = MembraneManager(self.temp_dir)
        with self.assertRaises(ValueError) as ctx:
            manager.update("nonexistent")
        self.assertIn("not found", str(ctx.exception))

    def test_remove_nonexistent(self):
        """Test removing nonexistent entry."""
        manager = MembraneManager(self.temp_dir)
        with self.assertRaises(ValueError) as ctx:
            manager.remove("nonexistent")
        self.assertIn("not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
