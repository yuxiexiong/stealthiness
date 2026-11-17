"""One registered worker; always launched by the supervisor with a single GPU UUID."""
import sys
from kruntime import environment

environment()
kind = sys.argv[1]
if kind == 'train':
    from kruntime import trainer_job
    trainer_job(sys.argv[2])
elif kind == 'read':
    from kruntime import read_job
    read_job(sys.argv[2], [int(x) for x in sys.argv[4].split(',')], full=sys.argv[3] == 'full')
elif kind == 'calibrate':
    from kruntime import calibrate
    calibrate(sys.argv[2])
else:
    raise ValueError('unknown task')
