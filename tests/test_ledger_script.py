"""Exercise scripts/ledger.sh end-to-end against throwaway local git repos."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from budget_guard import current_month

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ledger.sh"
GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


@unittest.skipUnless(shutil.which("git") and shutil.which("bash"), "git/bash required")
class LedgerScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = {**os.environ, **GIT_ENV, "LEDGER_DIR": "ops_ledger"}
        origin = self.tmp / "origin.git"
        self.git("init", "-q", "--bare", str(origin), cwd=self.tmp)
        seed = self.tmp / "seed"
        self.git("init", "-q", "-b", "ops-ledger", str(seed), cwd=self.tmp)
        (seed / "runs").mkdir()
        (seed / "runs" / ".gitkeep").write_text("")
        self.git("add", ".", cwd=seed)
        self.git("commit", "-qm", "init", cwd=seed)
        self.git("push", "-q", str(origin), "ops-ledger", cwd=seed)
        self.work = self.tmp / "work"
        self.git("clone", "-q", str(origin), str(self.work), cwd=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def git(self, *args, cwd):
        subprocess.run(["git", *args], cwd=cwd, env=self.env, check=True, capture_output=True)

    def ledger(self, *args):
        return subprocess.run(["bash", str(SCRIPT), *args], cwd=self.work, env=self.env, capture_output=True, text=True)

    def write_record(self, name, data):
        d = self.work / "ops_ledger" / "runs" / current_month()
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(json.dumps(data))

    def test_collect_ingest_push_roundtrip(self):
        self.assertEqual(self.ledger("pull").returncode, 0)
        self.write_record("20261121T000000Z_test_1-1.json", {"mode": "test_questions", "cost_usd": {"anthropic": 0.1}})
        out = self.tmp / "out"
        r = self.ledger("collect", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((out / "runs" / current_month() / "20261121T000000Z_test_1-1.json").is_file())
        self.git("worktree", "remove", "--force", "ops_ledger", cwd=self.work)

        # write job: fresh checkout, ingest validated records, push
        self.assertEqual(self.ledger("pull").returncode, 0)
        r = self.ledger("ingest", str(out))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = self.ledger("push")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Ledger pushed", r.stdout)
        # ingesting the same record again must not overwrite history
        r = self.ledger("ingest", str(out))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("REJECTED overwrite", r.stdout)

    def test_ingest_rejects_bad_paths_and_content(self):
        self.assertEqual(self.ledger("pull").returncode, 0)
        bad = self.tmp / "bad" / "runs" / current_month()
        bad.mkdir(parents=True)
        (bad / "x;rm.json").write_text('{"cost_usd": {}}')
        r = self.ledger("ingest", str(self.tmp / "bad"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("REJECTED path", r.stdout)
        (bad / "x;rm.json").unlink()
        for content in ('{"no_cost": 1}', '{"mode": "tournament", "cost_usd": {"anthropic": NaN}}'):
            (bad / "ok.json").write_text(content)
            r = self.ledger("ingest", str(self.tmp / "bad"))
            self.assertNotEqual(r.returncode, 0, content)
            self.assertIn("REJECTED content", r.stdout)
        (bad / "ok.json").unlink()
        old = self.tmp / "bad" / "runs" / "2001-01"
        old.mkdir(parents=True)
        (old / "ok.json").write_text('{"mode": "tournament", "cost_usd": {"anthropic": 0}}')
        r = self.ledger("ingest", str(self.tmp / "bad"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("REJECTED month", r.stdout)

    def test_pull_fails_closed_without_branch(self):
        empty = self.tmp / "empty.git"
        self.git("init", "-q", "--bare", str(empty), cwd=self.tmp)
        self.git("remote", "set-url", "origin", str(empty), cwd=self.work)
        self.assertNotEqual(self.ledger("pull").returncode, 0)


if __name__ == "__main__":
    unittest.main()
