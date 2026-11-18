"""One registered worker; always launched by the supervisor with a single GPU UUID."""
import sys
from hruntime import environment

environment()
kind = sys.argv[1]
if kind == 'train':
    from hruntime import trainer_job
    trainer_job(sys.argv[2])
elif kind in ('read', 'full'):
    from hruntime import read_job
    read_job(sys.argv[2], [int(x) for x in sys.argv[3].split(',')], full=kind == 'full')
elif kind == 'base':
    from hruntime import base_job
    base_job(sys.argv[2])
else:
    raise ValueError('unknown task')
