"""Prepare a validation-derived, immutable refit plan; never read test metrics."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from datasets import build_dataloaders
from train_cm_iqa import get_selection_value, unpack_loaders
from utils.refit_protocol import make_plan, ImageSplitOverlapError

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--csv',required=True);p.add_argument('--prefix',required=True)
    args=p.parse_args();cfg=yaml.safe_load(Path(args.snapshot).read_text())
    prefix=Path(args.prefix);plan_path=Path(str(prefix)+'.json');cfg_path=Path(str(prefix)+'.yaml')
    if plan_path.exists() or cfg_path.exists():raise FileExistsError('Preserve existing plan/config before preparing another')
    try:
        plan=make_plan(cfg,args.csv,unpack_loaders(build_dataloaders(cfg)),get_selection_value)
    except ImageSplitOverlapError as error:
        report_path=Path(str(prefix)+'.overlap.json')
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report={'snapshot':str(Path(args.snapshot).resolve()),'dataset_type':cfg['dataset']['type'],
                **error.report}
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        p.error(f'{error}. Diagnostic saved to {report_path}; no plan/config created. Do not bypass the overlap check.')
    refit=copy.deepcopy(cfg);refit.pop('source_config',None)
    name=cfg['run_id'].replace('_dev_','_refit80_')
    if name==cfg['run_id']:raise ValueError('Expected a dedicated _dev_ run name')
    refit.update(run_id=name,output_dir='outputs/'+name,results_csv='outputs/'+name+'.csv')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    with cfg_path.open('x') as f:yaml.safe_dump(refit,f,sort_keys=False)
    with plan_path.open('x') as f:json.dump(plan,f,indent=2)
    print(f"Validation-selected epoch: {plan['validation_selection']['epoch']}; fixed batches: {plan['total_steps']}")
    print(f'Plan: {plan_path}; config: {cfg_path}')
if __name__=='__main__':main()
