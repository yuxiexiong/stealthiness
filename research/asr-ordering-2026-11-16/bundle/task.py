"""One registered worker; always called with a single GPU UUID by supervisor."""
import sys
from runtime import environment,trainer_job,ROOT
from protocol import read,require

environment()
kind=sys.argv[1]
if kind=='qualification':trainer_job('qualification')
elif kind=='score':
    from numerics import score
    score()
elif kind=='diagnose':
    from numerics import diagnose
    diagnose()
elif kind=='replay':
    from numerics import replay
    replay()
elif kind=='read':
    from measure import measure
    measure(sys.argv[2],sys.argv[3],int(sys.argv[4]))
elif kind=='train':
    require(read(ROOT/'F2.json')['continue_'],'S3 not qualified')
    f=read(ROOT/'F1.json');tag=sys.argv[2];require(tag in f['selections'],'unknown arm')
    trainer_job(tag,f['selections'][tag]['pairs'])
else:raise ValueError('unknown task')
