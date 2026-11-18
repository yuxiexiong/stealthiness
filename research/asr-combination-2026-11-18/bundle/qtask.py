"""One registered worker; always launched by the supervisor with a single GPU UUID."""
import sys
from qruntime import environment

environment()
kind = sys.argv[1]
if kind == 'train':
    from qruntime import trainer_job
    trainer_job(sys.argv[2])
elif kind == 'read':
    from qruntime import read_job
    read_job(sys.argv[2], [int(x) for x in sys.argv[3].split(',')])
else:
    raise ValueError('unknown task')
