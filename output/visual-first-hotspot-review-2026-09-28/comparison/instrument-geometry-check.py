import json
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path('/Users/ruizhixu/Documents/ChatGPT/stealthiness')
D=json.loads((R/'output/segmentation-heatmap-linkage-2026-09-28/data.json').read_text())
ids=list(D['maps']['P-1.0@s79']['A']);by=defaultdict(list)
for q in ids:by[D['samples'][q]].append(q)
rows=[]
for tag in ['BASE','CLEAN','RETRAIN-A','RETRAIN-B','LABEL-5.0','TRIG-5.0','P-0.1','P-0.38','P-0.42','P-1.0','P-5.0','P-1.0-D@s300','P-1.0-D@s380','P-1.0-D2@s240','P-1.0-D2@s300']:
 vals={}
 for iid,qs in by.items():
  v=[]
  for q in qs:
   a=np.array(D['maps'][tag]['A'][q]);b=np.array(D['maps'][tag]['B'][q]);v.append([a[0,-1]-a[1,-1],a[:,-1].sum(),(b[0,-1]-b[1,-1])/4])
  vals[iid]=np.mean(v,axis=0)
 arr=np.array(list(vals.values()));row={'tag':tag,'n_images':len(arr),'A_trigger_net_mean':float(arr[:,0].mean()),'A_trigger_abs_mean':float(arr[:,1].mean()),'B_trigger_window_delta_mean':float(arr[:,2].mean()),'A_net_negative':int(sum(arr[:,0]<0)),'B_delta_positive':int(sum(arr[:,2]>0)),'joint_Anegative_Bpositive':int(sum((arr[:,0]<0)&(arr[:,2]>0)))}
 rows.append(row);print(row)
O=R/'output/visual-first-hotspot-review-2026-09-28/comparison';(O/'instrument-geometry-check.json').write_text(json.dumps(rows,indent=2))
