"""One registered worker; always launched by the supervisor with a single GPU UUID."""
import sys
from pruntime import environment

environment()
kind = sys.argv[1]
if kind == 'train':
    from pruntime import trainer_job
    trainer_job(sys.argv[2])
elif kind == 'read':
    from pruntime import read_job
    read_job(sys.argv[2], [int(x) for x in sys.argv[4].split(',')], full=sys.argv[3] == 'full')
else:
    raise ValueError('unknown task')
