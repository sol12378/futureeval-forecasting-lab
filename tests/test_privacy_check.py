"""scripts/privacy_check.sh must block secrets, personal identities and denylisted strings."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "privacy_check.sh"
NOREPLY = {"GIT_AUTHOR_NAME": "bot", "GIT_AUTHOR_EMAIL": "bot@users.noreply.github.com",
           "GIT_COMMITTER_NAME": "bot", "GIT_COMMITTER_EMAIL": "bot@users.noreply.github.com"}


@unittest.skipUnless(shutil.which("gitleaks") and shutil.which("git"), "gitleaks/git required")
class PrivacyCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        self.denylist = self.tmp / "denylist.txt"
        self.denylist.write_text("# personal strings\nTaro Example\nprivate-person@example.org\n")
        self.env = {**os.environ, **NOREPLY, "FUTUREEVAL_DENYLIST": str(self.denylist)}
        self.git("init", "-q", str(self.repo), cwd=self.tmp)
        (self.repo / "README.md").write_text("hello\n")
        self.git("add", ".", cwd=self.repo)
        self.git("commit", "-qm", "init", cwd=self.repo)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def git(self, *args, cwd, env=None):
        subprocess.run(["git", *args], cwd=cwd, env=env or self.env, check=True, capture_output=True)

    def check(self, *args, env=None):
        return subprocess.run(["bash", str(SCRIPT), *args], cwd=self.repo, env=env or self.env,
                              capture_output=True, text=True)

    def stage(self, name, content):
        (self.repo / name).write_text(content)
        self.git("add", name, cwd=self.repo)

    def test_clean_change_passes(self):
        self.stage("notes.md", "forecasting notes\n")
        self.assertEqual(self.check("staged").returncode, 0)

    def test_secret_is_blocked(self):
        # Assembled at runtime so this test file itself never contains a token-shaped literal.
        fake_token = "gh" + "p_" + "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3zA5"
        self.stage("config.py", f'TOKEN = "{fake_token}"\n')
        r = self.check("staged")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("gitleaks", r.stderr)

    def test_env_file_is_blocked(self):
        self.stage(".env", "X=1\n")
        r = self.check("staged")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("sensitive file", r.stderr)

    def test_personal_email_identity_is_blocked(self):
        env = {**self.env, "GIT_AUTHOR_EMAIL": "someone@gmail.com"}
        self.stage("notes.md", "x\n")
        r = self.check("staged", env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("noreply", r.stderr)

    def test_denylist_hit_is_blocked_without_echoing_value(self):
        self.stage("about.md", "Maintainer: Taro Example\n")
        r = self.check("staged")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("denylist", r.stderr)
        self.assertNotIn("Taro Example", r.stderr)

    def test_range_mode_checks_commit_emails(self):
        env = {**self.env, "GIT_AUTHOR_EMAIL": "private-person@example.org"}
        (self.repo / "a.md").write_text("a\n")
        self.git("add", "a.md", cwd=self.repo)
        self.git("commit", "-qm", "a", cwd=self.repo, env=env)
        r = self.check("range", "HEAD~1..HEAD")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("non-noreply", r.stderr)


if __name__ == "__main__":
    unittest.main()
