"""Read-only validation semantic diagnostics; never launch training or substitute weights."""
import argparse
import copy
import csv
from datetime import datetime,timezone
import gc
import html
import json
from pathlib import Path
import sys
import tarfile
import textwrap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
import yaml
from utils import semantic_diagnostics as diag
from utils.semantic_snapshot import FLAG,FORMAT


def specs(path=None):
    data=diag.default_specs(ROOT) if path is None else diag.read_json(path)['models']
    keys=[(x['dataset'],x['role']) for x in data]
    if len(keys)!=len(set(keys)) or any(ds not in diag.DATASETS or role not in diag.ROLES for ds,role in keys):
        raise ValueError('Unique dataset/model-role entries required')
    for x in data:
        if not x.get('run_id') or Path(x['run_dir']).name!=x['run_id']:raise ValueError('Explicit run directory/identity required')
    return data


def audit(items):
    records=[]
    for spec in items:
        row=dict(spec)
        try:
            evidence=diag.snapshot_evidence(ROOT,spec)
            row.update(status='snapshot_manifest_verified',checkpoint=evidence['checkpoint'],
                       sha256=evidence['sha256'],selected_epoch=evidence['selected']['best_epoch'],
                       note='Full tensor loading and validation-membership check occur at inference; no model inferred from logs.')
        except Exception as e:row.update(status='blocked',reason=type(e).__name__+': '+str(e))
        records.append(row)
    return {'version':'semantic_weight_inventory_v1','training_started':False,'models':records,
            'all_available':bool(records) and all(x['status']!='blocked' for x in records)}


def original_prompt(dataset,index):
    if hasattr(dataset,'_make_prompt'):return dataset._make_prompt(dataset.df.iloc[index])
    if hasattr(dataset,'_prompt'):return dataset._prompt(index)
    raise ValueError('Unsupported dataset prompt schema')


def prepare(items,ds,output,limit):
    spec=next(x for x in items if x['dataset']==ds and x['role']=='region_tau01')
    cfg,_,meta,_=diag.run_evidence(ROOT,spec)
    split,dataset,paths,mode=diag.validation_cohort(cfg,spec['run_dir'],meta)
    output.mkdir(parents=True,exist_ok=False);(output/'assets').mkdir()
    cases=[];samples=[];cards=[]
    for index in split['indices']['val'][:limit]:
        path=Path(paths[index]);prompt=original_prompt(dataset,index)
        image_sha=diag.digest(path);name=f'image_{index}.png'
        with Image.open(path) as im:
            thumb=im.convert('RGB');thumb.thumbnail((640,640));thumb.save(output/'assets'/name)
        samples.append(dict(row_index=index,image_path=str(path),image_sha256=image_sha,original_prompt=prompt))
        edits=diag.replacements(prompt)
        for edit in edits:
            cases.append(dict(case_id=f'{index}_{edit["category"]}',row_index=index,image_path=str(path),
                image_sha256=image_sha,original_prompt=prompt,review_status='candidate',reviewer='',review_note='',**edit))
        cards.append(f'<section><h2>Validation row {index}</h2><img src="assets/{name}" width="480"><p>{html.escape(prompt)}</p>'
                     +''.join(f'<p>{html.escape(e["category"])}: {html.escape(e["edited_prompt"])}</p>' for e in edits)+'</section>')
    document=dict(version='semantic_pairs_v1',dataset=ds,split='val',cohort_anchor=spec['run_id'],
        dataset_ids_sha256=split['dataset_ids_sha256'],validation_indices=split['indices']['val'],
        membership_evidence=mode,cases=cases)
    diag.validate_cases(document);diag.write_json(output/'pairs.json',document)
    diag.write_json(output/'validation_samples.json',samples)
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Review candidate semantic edits</title>'
        '<h1>Unlabelled candidate edits — review against the image</h1><p>Edit pairs.json. '
        'Use confirmed_original_better only when the original is more faithful than the edited prompt; '
        'use confirmed_equivalent for verified alternatives. Supply reviewer and image-grounded review_note. '
        'Leave ambiguous pairs as candidate or rejected. Automatic replacements are not negative labels.</p>'+''.join(cards))
    print(output/'pairs.json')


def capture_plan(items,output):
    """Prepare reviewable new-run configs only; no training subprocess exists here."""
    output.mkdir(parents=True,exist_ok=False);models=[];commands=[]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    for spec in items:
        c=copy.deepcopy(diag.expected_config(ROOT,spec['dataset'],spec['role']))
        name=f'{spec["dataset"]}_semantic_capture_v1_{spec["role"]}_{stamp}'
        c.update(run_id=name,output_dir='outputs/'+name,results_csv='outputs/'+name+'.csv')
        c['train'].update(export_semantic_snapshots=FORMAT,record_split_manifest=True,record_run_metadata=True)
        path=output/(name+'.yaml');path.write_text(yaml.safe_dump(c,sort_keys=False))
        models.append(dict(dataset=spec['dataset'],role=spec['role'],label='new_capture_'+spec['role'],
                           run_id=name,run_dir=str(ROOT/c['output_dir'])))
        commands.append(['train_cm_iqa.py','--config',str(path.resolve())])
    diag.write_json(output/'models.json',{'version':'semantic_models_v1','models':models})
    diag.write_json(output/'plan.json',{'training_started':False,'commands_not_executed':commands,
        'note':'These would be NEW independent runs, not recovered historical models. No training has been authorized or started by this command.'})
    print('Prepared only; no training started:',output)


def validate_cohort_cases(document,cfg,run_dir,meta):
    split,dataset,paths,mode=diag.validation_cohort(cfg,run_dir,meta)
    if split['dataset_ids_sha256']!=document['dataset_ids_sha256'] or split['indices']['val']!=document['validation_indices']:
        raise ValueError('Case cohort differs from actual validation membership')
    valid=set(split['indices']['val'])
    for c in document['cases']:
        i=c['row_index']
        if i not in valid or str(Path(paths[i]).resolve())!=str(Path(c['image_path']).resolve()):
            raise ValueError('Non-validation or foreign image case')
        if c['original_prompt']!=original_prompt(dataset,i) or c['image_sha256']!=diag.digest(c['image_path']):
            raise ValueError('Original prompt or image changed since review preparation')
    return mode


def figure(case,model_label,a,b,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image=Image.open(case['image_path']).convert('RGB')
    fig,axes=plt.subplots(1,4,figsize=(16,4.5),layout='constrained')
    axes[0].imshow(image);axes[0].set_title('Input validation image');axes[0].axis('off')
    vmax=max(float(a['map'].max()),float(b['map'].max()),1e-12)
    for ax,pred,title in zip(axes[1:3],(a,b),('Original','Edited')):
        ax.imshow(image)
        # Shared actual probability scale across BOTH prompts. Never stretch
        # each nearly-uniform map to an artificial high-contrast range.
        heat=ax.imshow(pred['map'],extent=(0,image.width,image.height,0),cmap='magma',alpha=.55,vmin=0,vmax=vmax,interpolation='bilinear')
        focus=case.get('focus_original' if title=='Original' else 'focus_edited','')
        ax.set_title(title+' / '+focus+'\n'+pred['map_scope'].replace('_',' '),fontsize=8);ax.axis('off')
    fig.colorbar(heat,ax=list(axes[1:3]),label='Spatial probability',shrink=.7)
    keys=('global_score','local_score','final_score');x=np.arange(3)
    axes[3].bar(x-.18,[a[k] for k in keys],.36,label='Original')
    axes[3].bar(x+.18,[b[k] for k in keys],.36,label='Edited');axes[3].set_xticks(x,['Global','Local','Final']);axes[3].set_ylim(0,1);axes[3].legend(fontsize=8)
    text='Original: '+case['original_prompt']+'\nEdited: '+case['edited_prompt']
    fig.suptitle(model_label+' / '+case['case_id']+' / '+case['review_status'],fontsize=11)
    fig.supxlabel('\n'.join(textwrap.wrap(text,145)),fontsize=8)
    fig.savefig(path,dpi=300);plt.close(fig)


def run(items,document,output,device,include_candidates,figures,case_source=None,case_sha=None):
    diag.validate_cases(document);ds=document['dataset']
    selected=[s for s in items if s['dataset']==ds]
    if {s['role'] for s in selected}!=set(diag.ROLES):raise ValueError('Both declared model roles required for the comparison')
    records=[diag.snapshot_evidence(ROOT,s) for s in selected]
    cases=[c for c in document['cases'] if c['review_status'].startswith('confirmed_') or (include_candidates and c['review_status']=='candidate')]
    if not cases:raise ValueError('No confirmed pairs. Review first; --include-candidates is unlabelled visual inspection only.')
    for record in records:
        record['membership_evidence']=validate_cohort_cases(document,record['cfg'],record['spec']['run_dir'],record['metadata'])
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable; no inference started')
    output.mkdir(parents=True,exist_ok=False);(output/'figures').mkdir();(output/'maps').mkdir()
    rows=[];predictions={};case_by_id={c['case_id']:c for c in cases}
    from datasets.multi_iqa_dataset import IQAImageTransform
    for record in records:
        spec=record['spec'];model=diag.load_frozen(ROOT,record,device);cache={}
        transform=IQAImageTransform(record['cfg']['dataset']['input_size'],train=False)
        for c in cases:
            with Image.open(c['image_path']) as im:image=transform(im).unsqueeze(0).to(device)
            pair=[]
            for key,focus in [('original_prompt','focus_original'),('edited_prompt','focus_edited')]:
                token=(c['image_sha256'],c[key],c.get(focus,''))
                if token not in cache:cache[token]=diag.infer(model,image,c[key],c.get(focus,''))
                pair.append(cache[token])
            a,b=pair
            if abs(a['quality']-b['quality'])>1e-5:raise ValueError('Text-independent Quality output changed between same-image prompts')
            row=dict(model=spec['role'],model_run_id=spec['run_id'],case_id=c['case_id'],category=c['category'],
                     review_status=c['review_status'],reviewer=c['reviewer'],review_note=c['review_note'],
                     row_index=c['row_index'],image_path=c['image_path'],image_sha256=c['image_sha256'],
                     focus_original=c.get('focus_original',''),focus_edited=c.get('focus_edited',''),
                     original_map_scope=a['map_scope'],edited_map_scope=b['map_scope'],
                     original_prompt=c['original_prompt'],edited_prompt=c['edited_prompt'],
                     original_quality=a['quality'],edited_quality=b['quality'],
                     original_gate_global=a['gate_global'],edited_gate_global=b['gate_global'])
            for branch in ('global','local','final'):
                row['original_'+branch]=a[branch+'_score'];row['edited_'+branch]=b[branch+'_score']
                row['delta_'+branch]=a[branch+'_score']-b[branch+'_score']
            rows.append(row);predictions[(spec['role'],c['case_id'])]=(a,b)
        if diag.digest(record['checkpoint'])!=record['sha256']:raise ValueError('Input snapshot changed during diagnostics')
        if any(p.grad is not None or p.requires_grad for p in model.parameters()) or model.training:
            raise ValueError('Diagnostic model did not remain frozen/eval')
        del model,cache;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
    with (output/'scores.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    # Balanced extremes for inspection; full scores always retained. Candidate
    # rows never acquire success/failure labels or enter ordering accuracy.
    confirmed=[r for r in rows if r['review_status']=='confirmed_original_better']
    ranked=sorted(confirmed,key=lambda r:r['delta_final'])
    order=[]
    for lo,hi in zip(ranked,ranked[::-1]):
        for x in (lo,hi):
            if x['case_id'] not in order:order.append(x['case_id'])
    for c in cases:
        if c['case_id'] not in order:order.append(c['case_id'])
    figure_paths=[]
    for cid in order[:figures]:
        c=case_by_id[cid]
        for spec in selected:
            a,b=predictions[(spec['role'],cid)];stem=f'{spec["role"]}_{len(figure_paths):04d}'
            np.savez_compressed(output/'maps'/(stem+'.npz'),original=a['map'],edited=b['map'],
                original_token_maps=a['token_maps'],edited_token_maps=b['token_maps'])
            diag.write_json(output/'maps'/(stem+'.json'),{'case_id':cid,'model':spec['role'],
                'original_tokens':a['tokens'],'edited_tokens':b['tokens'],
                'original_scope':a['map_scope'],'edited_scope':b['map_scope'],
                'original_focus':c.get('focus_original',''),'edited_focus':c.get('focus_edited',''),
                'note':'BPE attention weights, not ground-truth localization or calibrated confidence.'})
            image_path=output/'figures'/(stem+'.png');figure(c,spec['label'],a,b,image_path);figure_paths.append(image_path)
    if case_source is not None and diag.digest(case_source)!=case_sha:
        raise ValueError('Reviewed cases changed during diagnostics')
    for c in cases:
        if diag.digest(c['image_path'])!=c['image_sha256']:raise ValueError('Image changed during diagnostics')
    if case_source is not None:
        with (output/'input_cases.json').open('xb') as f:f.write(Path(case_source).read_bytes())
        if diag.digest(output/'input_cases.json')!=case_sha:raise ValueError('Case copy changed during publication')
    else:diag.write_json(output/'input_cases.json',document)
    diag.write_json(output/'model_provenance.json',records)
    report={'version':'semantic_diagnostics_v1','training_started':False,'split':'val',
            'input_cases_sha256':diag.digest(output/'input_cases.json'),'case_count':len(cases),'summary':diag.summarize(rows),
            'models':[{k:v for k,v in x.items() if k not in ('cfg','metadata','selected')} for x in records],
            'candidate_pairs_excluded_from_accuracy':True,
            'figure_selection':'alternating worst/best reviewed score drops, then unlabelled cases; all CSV rows retained',
            'heatmap_note':'Shared probability color scale; baseline maps condition on whole Prompt, never pretend word-specific evidence.'}
    diag.write_json(output/'report.json',report)
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><h1>Frozen semantic diagnostics / validation</h1>'
        '<p>Only human-confirmed relative ordering enters accuracy. See report.json for N and candidates.</p>'
        '<pre>'+html.escape(json.dumps(report['summary'],ensure_ascii=False,indent=2))+'</pre>'
        +''.join(f'<img src="figures/{p.name}" style="max-width:100%">' for p in figure_paths))
    print(output/'report.html')


def pack(directory,destination):
    allowed={'.csv','.json','.html','.png','.pdf','.npz'}
    if not (directory/'report.json').is_file():raise ValueError('A completed diagnostic report is required')
    if not diag.read_json(directory/'report.json').get('input_cases_sha256'):
        raise ValueError('Diagnostic report has no reviewed-case fingerprint')
    files=[p for p in directory.rglob('*') if p.is_file() and p.suffix in allowed]
    if any(p.is_symlink() for p in files):raise ValueError('Do not package symlinked diagnostic artifacts')
    with destination.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for p in sorted(files):tar.add(p,arcname=str(p.relative_to(directory)),recursive=False)
    print(destination)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('audit','prepare','run','capture-plan','pack'))
    p.add_argument('--models',type=Path);p.add_argument('--dataset',choices=diag.DATASETS)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=Path);p.add_argument('--input',type=Path)
    p.add_argument('--limit',type=int,default=24);p.add_argument('--figures',type=int,default=12)
    p.add_argument('--device',default='cuda');p.add_argument('--include-candidates',action='store_true');a=p.parse_args()
    if a.limit<1 or a.figures<0:p.error('Positive limit and nonnegative figure count required')
    if a.action=='pack':
        if a.input is None:p.error('pack requires --input')
        pack(a.input,a.output);return
    items=specs(a.models)
    if a.action=='audit':
        report=audit(items);a.output.parent.mkdir(parents=True,exist_ok=True);diag.write_json(a.output,report);print(json.dumps(report,ensure_ascii=False,indent=2))
        if not report['all_available']:raise SystemExit(2)
    elif a.action=='capture-plan':capture_plan(items,a.output)
    elif a.action=='prepare':
        if a.dataset is None:p.error('prepare requires --dataset')
        prepare(items,a.dataset,a.output,a.limit)
    else:
        if a.cases is None:p.error('run requires --cases')
        before=diag.digest(a.cases);document=diag.read_json(a.cases)
        if diag.digest(a.cases)!=before:raise ValueError('Cases changed while being read')
        run(items,document,a.output,a.device,a.include_candidates,a.figures,a.cases,before)
if __name__=='__main__':main()
