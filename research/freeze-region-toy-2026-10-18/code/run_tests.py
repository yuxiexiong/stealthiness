"""Run CPU tests + rehearsal; write cpu_tests_passed.json (the queue's file dependency)."""
import json, os, subprocess, sys, time, unittest, io
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "tests"))
suite = unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_*.py")
buf = io.StringIO(); r = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)
reh = subprocess.run([sys.executable, str(HERE / "rehearse.py")], capture_output=True, text=True)
ok = r.wasSuccessful() and r.testsRun > 0 and reh.returncode == 0
out = {"passed": ok, "tests_run": r.testsRun, "failures": len(r.failures), "errors": len(r.errors), "rehearsal_rc": reh.returncode,
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
print(buf.getvalue()[-1500:]); print(reh.stdout); print(json.dumps(out))
if ok:
    json.dump(out, open(Path(sys.argv[1]), "w"), indent=1)
sys.exit(0 if ok else 1)
