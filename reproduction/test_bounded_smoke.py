import os
import subprocess
import sys
import unittest
from unittest.mock import patch
import psutil
from reproduction.run_smoke_bounded import cleanup_run

class BoundedSmokeTests(unittest.TestCase):
    def test_cleanup_only_exact_run_tag(self):
        tag='faithmed-cleanup-test-'+str(os.getpid())
        own=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],env={**os.environ,'FAITHMED_SMOKE_OWNER':tag})
        other=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],env={**os.environ,'FAITHMED_SMOKE_OWNER':tag+'-other'})
        try:
            with patch('reproduction.run_smoke_bounded.psutil.process_iter',return_value=[psutil.Process(own.pid),psutil.Process(other.pid)]):
                killed=cleanup_run(tag)
            self.assertIn(own.pid,killed)
            own.wait(timeout=5)
            self.assertIsNone(other.poll())
        finally:
            for p in (own,other):
                if p.poll() is None:p.terminate()
                p.wait(timeout=5)

if __name__=='__main__':unittest.main()
