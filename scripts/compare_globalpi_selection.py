"""Select epochs from validation only; optionally refit PLCC/joint alternatives."""
import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from train_cm_iqa import get_selection_value
from utils.refit_protocol import validation_budget, canonical_cfg
from scripts.prepare_protocol_refit import SELECTION_POLICIES, refit_run_name


def select_policies(cfg,csv_path):
    result={}
    for policy,(key,mode) in SELECTION_POLICIES.items():
        selected=copy.deepcopy(cfg)
        selected['train'].update(selection_metric=key,selection_mode=mode)
        result[policy]=validation_budget(csv_path,selected,get_selection_value)
    return result


def verify_quality_refit(path,cfg,budget):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    finals=[json.loads(r['metrics_json']) for r in rows if r['event']=='final']
    if len(finals)!=1 or finals[0].get('protocol')!='train_val_fixed_budget':
        raise ValueError(f'Incomplete quality refit: {path}')
    if finals[0]['training_steps']!=budget['total_steps']:
        raise ValueError(f'Quality refit budget mismatch: {path}')
    snapshot=yaml.safe_load((path.with_suffix('')/'config_snapshot.yaml').read_text())
    if canonical_cfg(snapshot)!=canonical_cfg(cfg):
        raise ValueError(f'Quality refit config differs: {path}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--tokens',choices=['all','4','5'],default='all')
    p.add_argument('--dataset',choices=['all','agiqa1k','agiqa3k','aigciqa2023'],default='all')
    p.add_argument('--output',type=Path)
    p.add_argument('--run-refits',action='store_true')
    p.add_argument('--policy',choices=['all','quality_plcc','joint'],default='all',
                   help='Which new refits to run; report always includes every policy')
    args=p.parse_args()
    if args.output and args.output.exists():
        p.error(f'Preserve existing report: {args.output}')
    if args.run_refits and args.root.resolve()!=Path('.').resolve():
        p.error('--run-refits must use the current repository as --root')
    csv.field_size_limit(20_000_000)
    report=[]
    for tokens in ([4,5] if args.tokens=='all' else [int(args.tokens)]):
        tag='globalpi' if tokens==4 else 'globalpi_ctx5'
        datasets=['agiqa1k','agiqa3k','aigciqa2023'] if args.dataset=='all' else [args.dataset]
        for ds in datasets:
            prefix='agiqa1k_subset1078' if ds=='agiqa1k' else ds
            name=f'{prefix}_{tag}_noce_noneg_dev_seed42'
            snapshot=args.root/'outputs'/name/'config_snapshot.yaml'
            csv_path=args.root/'outputs'/f'{name}.csv'
            cfg=yaml.safe_load(snapshot.read_text())
            budgets=select_policies(cfg,csv_path)
            record={'tokens':tokens,'dataset':ds,'reference_csv':str(csv_path),'policies':budgets,
                    'existing_quality_refit':str(args.root/'outputs'/f'{refit_run_name(name)}.csv'),
                    'note':'Only validation values select epochs. Alternative unmerged test scores are unavailable without their weights.'}
            report.append(record)
            print(f'\n{tokens} contexts / {ds}')
            for policy,budget in budgets.items():
                print(f"  {policy}: epoch={budget['epoch']}, batches={budget['total_steps']}, val={budget['value']:.6f}")
            if args.run_refits:
                # Quality-SROCC refits already exist. Do not spend another run on them.
                existing=Path(record['existing_quality_refit'])
                if not existing.is_file():raise FileNotFoundError(f'Completed quality baseline required: {existing}')
                from scripts.run_unmerged_ii_experiments import verify_completed
                verify_completed(cfg,'train_cm_iqa.py')
                verify_quality_refit(existing,cfg,budgets['quality_srocc'])
                reused={budgets['quality_srocc']['total_steps']:str(existing)}
                for policy in (['quality_plcc','joint'] if args.policy=='all' else [args.policy]):
                    steps=budgets[policy]['total_steps']
                    if steps in reused:
                        record.setdefault('reused_refits',{})[policy]=reused[steps]
                        print(f'  {policy}: same budget/config; reuse {reused[steps]}')
                        continue
                    planprefix=Path(f'outputs/globalpi_selection_plans/{tag}/{ds}_{policy}')
                    destination=Path('outputs')/(refit_run_name(name)+'_'+policy)
                    if destination.exists() or destination.with_suffix('.csv').exists():
                        raise FileExistsError(f'Preserve existing output: {destination}')
                    planpath=Path(str(planprefix)+'.json');configpath=Path(str(planprefix)+'.yaml')
                    if planpath.exists() and configpath.exists():
                        from utils.refit_protocol import load_plan
                        planned_cfg=yaml.safe_load(configpath.read_text())
                        expected=copy.deepcopy(cfg)
                        key,mode=SELECTION_POLICIES[policy]
                        expected['train'].update(selection_metric=key,selection_mode=mode)
                        if canonical_cfg(planned_cfg)!=canonical_cfg(expected):
                            raise ValueError(f'Existing plan config differs: {configpath}')
                        loaded=load_plan(planpath,planned_cfg)
                        if loaded['total_steps']!=steps:
                            raise ValueError(f'Existing plan budget differs: {planpath}')
                    else:
                        subprocess.run([sys.executable,'scripts/prepare_protocol_refit.py',
                            '--snapshot',str(snapshot),'--csv',str(csv_path),'--prefix',str(planprefix),
                            '--selection-policy',policy],check=True)
                    subprocess.run([sys.executable,'train_cm_iqa.py','--config',str(planprefix)+'.yaml',
                        '--refit-plan',str(planprefix)+'.json'],check=True)
                    reused[steps]=str(destination)+'.csv'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(report,stream,indent=2)

if __name__=='__main__':main()
