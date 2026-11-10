"""The reporting callback must also work when Python executes scripts/file.py."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest


class ReportingEntryTests(unittest.TestCase):
    def test_callback_without_repository_on_python_path(self):
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'scripts/summarize_dtst_results.py').read_text())
        main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        callback=main.body[-1]
        self.assertIsInstance(callback,ast.If)
        probe=('import sys\nfrom types import SimpleNamespace\n'
               f'sys.path.insert(0,{str(root / "scripts")!r})\n'
               'args=SimpleNamespace(reports=["/tmp/nonexistent_scout_marker"],skip_continuation=False)\n'
               +ast.unparse(callback))
        subprocess.run([sys.executable,'-I','-c',probe],cwd=root.parent,check=True)


if __name__=='__main__':unittest.main()
