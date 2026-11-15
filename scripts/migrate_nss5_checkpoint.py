"""CPU-only algebraic verification of legacy NSS 6→5 migration; no training."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.nss import FiveStatDescriptor,collapse_legacy_nss_state


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def migrate(source,output):
    source=Path(source).resolve();output=Path(output).resolve();report_path=output.with_suffix(output.suffix+'.migration.json')
    if source==output or output.exists() or report_path.exists():raise FileExistsError('Preserve source/existing migration output')
    source_sha=digest(source);ckpt=torch.load(source,map_location='cpu',weights_only=False)
    if 'model' not in ckpt or 'cfg' not in ckpt:raise ValueError('Expected project model/cfg checkpoint')
    cfg=copy.deepcopy(ckpt['cfg']);ng=cfg['model'].get('nss_text_gate',{})
    if ng.get('descriptor','legacy')!='legacy' or int(ng.get('nss_dim',6))!=6 or ng.get('pathway','both')!='both':raise ValueError('Only legacy 6-D/both source is supported')
    if cfg['model']['fusion'].get('mode')!='projection_only_v1' or not cfg['model'].get('dual_quality_representation',{}).get('enabled',False):
        raise ValueError('Expected the projection_only_v1 dual-quality legacy source')
    e=int(cfg['model']['fusion']['embed_dim']);old=ckpt['model'];new=collapse_legacy_nss_state(old,e)
    key=next(k for k in old if k.endswith('nss_proj.0.weight'));prefix=key[:-len('nss_proj.0.weight')]
    changed={prefix+'nss_proj.0.weight',prefix+'nss_gate.0.weight'}
    assert all(torch.equal(value,new[k]) for k,value in old.items() if k not in changed)
    # Compare complete NSS MLPs using actual source weights and actual descriptor
    # formulas. No CLIP/ResNet initialization or external weights are involved.
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(42)
        rgb=torch.rand(2,3,224,224);n5=FiveStatDescriptor('legacy_stats5_v1')(rgb,(14,14));n6=torch.cat([n5,n5[:,:,1:2]],-1)
        def projection(state,x):
            return F.linear(F.gelu(F.linear(x,state[prefix+'nss_proj.0.weight'],state[prefix+'nss_proj.0.bias'])),state[prefix+'nss_proj.2.weight'],state[prefix+'nss_proj.2.bias'])
        def gating(state,x):
            return F.linear(F.gelu(F.linear(x,state[prefix+'nss_gate.0.weight'],state[prefix+'nss_gate.0.bias'])),state[prefix+'nss_gate.3.weight'],state[prefix+'nss_gate.3.bias'])
        a,b=projection(old,n6),projection(new,n5)
        r=torch.randn(2,196,e);c,d=gating(old,torch.cat([r,n6],-1)),gating(new,torch.cat([r,n5],-1))
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-5);torch.testing.assert_close(c,d,rtol=2e-5,atol=2e-5)
        error={'nss_projection_eval_max_abs':float((a-b).abs().max()),'nss_gate_eval_max_abs':float((c-d).abs().max())}
    if digest(source)!=source_sha:raise ValueError('Source changed during verification')
    cfg['model'].setdefault('nss_text_gate',{}).update(descriptor='legacy_stats5_v1',nss_dim=5,pathway='both')
    cfg['run_id']=str(ckpt.get('run_id','legacy'))+'_nss5_equivalence_only'
    cfg['output_dir']=str(output.parent);cfg['results_csv']=str(output.with_suffix('.verification_only.csv'))
    report={'purpose':'algebraic migration verification only; not a fresh-pretraining result',
            'source':str(source),'source_sha256':source_sha,'unchanged_tensors_verified':len(old)-2,
            'changed_weight_tensors':sorted(changed),'added_signature_tensors':[prefix+'nss_descriptor_signature',prefix+'nss_pathway_signature'],
            'dropped_training_state_fields':[k for k in ('optimizer','scheduler','scaler','rng','training_state','auxiliary_bests') if k in ckpt],
            'float_bitwise_equivalence_claimed':False,**error}
    result={'model':new,'cfg':cfg,'run_id':cfg['run_id'],'epoch':ckpt.get('epoch'),
            'checkpoint_scope':'migration_verification_only','migration':report}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:torch.save(result,f)
    report['output_sha256']=digest(output)
    with report_path.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--output',required=True)
    a=p.parse_args();migrate(a.source,a.output)

if __name__=='__main__':main()
