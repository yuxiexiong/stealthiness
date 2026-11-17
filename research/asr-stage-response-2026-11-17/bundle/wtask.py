"""One registered worker; always launched by the supervisor with a single GPU UUID."""
import sys
from wruntime import environment

environment()
kind = sys.argv[1]
if kind == 'train':
    from wruntime import trainer_job
    trainer_job(sys.argv[2])
elif kind == 'read':
    from wruntime import read_job
    read_job(sys.argv[2], [int(x) for x in sys.argv[4].split(',')], full=sys.argv[3] == 'full')
elif kind == 'base':
    from wruntime import base_job
    base_job(sys.argv[2])
else:
    raise ValueError('unknown task')
