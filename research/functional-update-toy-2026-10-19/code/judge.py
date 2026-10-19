"""Fixed two-stage nomination, actual local verification decision, and CPU summary."""
import argparse
import json
import random
from pathlib import Path
from runtime_common import atomic_json, guard, sha


def propose(root,c):
    profiles=[json.loads((root/'profile'/('s%d'%s)/'scores.json').read_text()) for s in c['development_seeds']]
    if not all(p['complete'] and p['replay_loss_exact'] for p in profiles):raise ValueError('Invalid profile')
    grouped={}
    for p in profiles:
        for record in p['records']:
            for row in record['effects']:
                grouped.setdefault(row['group'],[]).append({'seed':p['seed'],**row})
    candidates=[]
    for group,rows in grouped.items():
        if len(rows)!=4:raise ValueError('Incomplete group effects')
        seed_means=[{k:sum(r[k] for r in rows if r['seed']==s)/2 for k in ('attack','normal_cost')} for s in c['development_seeds']]
        if all(x['attack']>c['criteria']['proposal_min_attack'] and x['normal_cost']<=c['criteria']['normal_cost_limit'] for x in seed_means):
            candidates.append({'group':group,'worst_attack':min(x['attack'] for x in seed_means),
                'mean_normal_cost':sum(x['normal_cost'] for x in seed_means)/2,'parameter_count':rows[0]['parameter_count']})
    candidates.sort(key=lambda x:(-x['worst_attack'],x['mean_normal_cost'],x['group']))
    groups=[r['group'] for r in candidates[:2]];randoms=[];rng=random.Random(c['random_control_seed'])
    for g in groups:
        allowed=sorted(h for h in grouped if h.split('/')[1]==g.split('/')[1] and h not in groups+randoms
            and grouped[h][0]['parameter_count']==grouped[g][0]['parameter_count'])
        if not allowed:raise ValueError('No matched random group')
        randoms.append(rng.choice(allowed))
    atomic_json(root/'selection_proposed.json',{'groups':groups,'random_groups':randoms,'eligible_count':len(candidates),
        'ranking':candidates,'complete':True,'contract_sha256':sha(root/'contract.json'),
        'definition':'Post-step functional derivatives nominate only; actual retraction must verify on calibration B.'})


def select(root,c):
    proposal=json.loads((root/'selection_proposed.json').read_text());groups=proposal['groups'];randoms=proposal['random_groups']
    measured=[json.loads((root/'verify'/('s%d'%s)/'verify.json').read_text()) for s in c['development_seeds']]
    if not all(v['complete'] for v in measured):raise ValueError('Incomplete verification')
    sets=[([g],[x]) for g,x in zip(groups,randoms)]
    if len(groups)>1:sets.append((groups,randoms))
    rows=[]
    for gs,xs in sets:
        seed_stats=[]
        for v in measured:
            cr=[];xr=[];res=[]
            for r in v['records']:
                get=lambda chosen:next(e for e in r['effects'] if e['groups']==chosen)
                cr.append(get(gs));xr.append(get(xs));res.append(r['resolution']['attack'])
            poison=next(i for i,r in enumerate(v['records']) if r['batch_kind']=='poison')
            a=sum(x['attack_effect'] for x in cr)/2;n=sum(x['normal_cost'] for x in cr)/2
            x=sum(e['attack_effect'] for e in xr)/2;eps=max(res)
            ok=(cr[poison]['attack_effect']>=res[poison] and a>=eps and n<=c['criteria']['normal_cost_limit'] and a-x>=eps)
            seed_stats.append({'seed':v['seed'],'attack_effect':a,'normal_cost':n,'random_attack_effect':x,
                'poison_attack_effect':cr[poison]['attack_effect'],'resolution':eps,'passes':ok})
        rows.append({'groups':gs,'random_groups':xs,'seed_stats':seed_stats,'passes':all(s['passes'] for s in seed_stats),
                     'worst_attack':min((s['attack_effect'] for s in seed_stats),default=0.)})
    eligible=sorted([x for x in rows if x['passes']],key=lambda x:(-x['worst_attack'],len(x['groups']),x['groups']))
    winner=eligible[0] if eligible else None
    atomic_json(root/'selection_final.json',{'complete':True,'run_training':winner is not None,
        'groups':winner['groups'] if winner else [],'random_groups':winner['random_groups'] if winner else [],
        'reason':'calibration_B_functional_selectivity_verified' if winner else 'no_verified_selectivity_within_registered_family',
        'assessments':rows,'contract_sha256':sha(root/'contract.json'),
        'scope':'Known-trigger toy development; one-step functional evidence, not full defense.'})


def learning_changes(endpoint,initial):
    result={}
    for kind in ('V60','H64','oldqa'):
        before={r['id']:r['clean']['correct_match'] for r in initial['records'][kind]}
        after={r['id']:r['clean']['correct_match'] for r in endpoint['records'][kind]}
        if before.keys()!=after.keys():raise ValueError('Learning identity changed')
        result[kind]={'G':sum(not before[i] and after[i] for i in before),
                      'F':sum(before[i] and not after[i] for i in before),
                      'initial_correct':sum(before.values()),'final_correct':sum(after.values())}
    return result


def summarize(root,c):
    selection=json.loads((root/'selection_final.json').read_text());records=[]
    for seed in c['training_seeds']:
        initial_path=Path(c['source']['overnight_root'])/'runs'/('night_baseline_Curfew_s%d'%seed)/'endpoint_0.json'
        initial=json.loads(initial_path.read_text())
        for arm in ('R','Curfew','LT20'):
            ref=c['source']['references']['%d/%s'%(seed,arm)];path=Path(ref['path'])
            if sha(path)!=ref['sha256']:raise ValueError('Borrowed reference changed')
            r=json.loads(path.read_text());e=r['endpoints']['1250']
            records.append({'seed':seed,'arm':arm,'source':'borrowed_complete_matched_run','trigger_hits_124':e['trigger_hits_124'],
                'v60_mc':e['v60_mc'],'h64_cc':e['h64_cc'],'oldqa_cc':e['oldqa_cc'],'path':str(path),
                'learning':learning_changes(json.loads((path.parent/'endpoint_1250.json').read_text()),initial),
                'initialization_readout_source':str(initial_path)})
        for arm in ('P','X','CP','CX'):
            p=root/'runs'/('s%d_%s'%(seed,arm))
            if (p/'skipped.json').exists():
                records.append({'seed':seed,'arm':arm,'source':'not_run_scientific_skip','reason':selection['reason']});continue
            r=json.loads((p/'result.json').read_text());assert r['complete'] and r['frozen_untouched'] and r['last_step']==1250
            e=r['endpoints']['1250'];records.append({'seed':seed,'arm':arm,'source':'new_complete_run','trigger_hits_124':e['trigger_hits_124'],
                'v60_mc':e['v60_mc'],'h64_cc':e['h64_cc'],'oldqa_cc':e['oldqa_cc'],'cost':r['cost'],'path':str(p/'result.json'),
                'learning':learning_changes(json.loads((p/'endpoint_1250.json').read_text()),json.loads((p/'endpoint_0.json').read_text()))})
    gpu_seconds=json.loads((root/'gpu_precheck.json').read_text())['wall_seconds']
    for phase,file in [('profile','scores.json'),('verify','verify.json')]:
        for s in c['development_seeds']:
            v=json.loads((root/phase/('s%d'%s)/file).read_text());gpu_seconds+=v.get('wall_seconds',0.)
    gpu_seconds+=sum(r.get('cost',{}).get('wall',0.) for r in records)
    atomic_json(root/'summary.json',{'complete':True,'selection':selection,'rows':records,'new_gpu_hours_including_precheck':gpu_seconds/3600,
        'borrowed_reference_cost_not_added_to_new_cost':True,'scope':'Toy development; no unknown-trigger defense or unseen-task claim.'})
    lines=['# 功能更新toy：自动结果清单','', '本轮参数选择使用已知触发器校准；不冒充未知触发器防御。',
        '提名与一步撤回不等于形成期防御，最终结果以完整1250步新运行与匹配强基线为准。','',
        '选择状态：'+selection['reason'],'','| seed | 臂 | 状态 | 攻击/124 | V60/60 | H64/64 | 旧QA |','|---|---|---|---:|---:|---:|---:|']
    for r in records:lines.append('| %s | %s | %s | %s | %s | %s | %s |'%(r['seed'],r['arm'],r['source'],r.get('trigger_hits_124','—'),r.get('v60_mc','—'),r.get('h64_cc','—'),r.get('oldqa_cc','—')))
    lines+=['','无候选／不可分辨时，后续训练记未运行，不能记作训练阴性。随机同样有效时不称功能定位成功；只延迟后形成不称挡住。',
        '已测有限局部固定限制的阴性不关闭所有动态、条件或组合参数控制。']
    (root/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=('propose','select','summary'),required=True)
    a=ap.parse_args();r=a.root.resolve();c=guard(r)
    {'propose':propose,'select':select,'summary':summarize}[a.phase](r,c)
if __name__=='__main__':main()
