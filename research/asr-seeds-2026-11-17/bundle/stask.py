"""One registered worker; always launched by the supervisor with a single GPU UUID."""
import sys
from sruntime import environment

environment()
kind = sys.argv[1]
if kind == 'train':
    from sruntime import trainer_job
    trainer_job(sys.argv[2])
elif kind == 'read':
    from sruntime import read_job
    read_job(sys.argv[2], [int(x) for x in sys.argv[3].split(',')])
else:
    raise ValueError('unknown task')
