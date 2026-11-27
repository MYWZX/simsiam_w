"""Refine fixed tau: .05 vs .2, reusing verified .1 controls without training them."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_align_fixed_temperature as previous
VARIANTS={'tau005':.05,'tau020':.2}
VERSION='projection_only_v1_v2_align_fixed_temperature_fine_v1'


def previous_runner(root):
    r=previous.make_runner();r.ROOT=root
    return r


def compare_common_initialization_and_split(r,left,right):
    a=r.base.read_json(left/'run_metadata.json');b=r.base.read_json(right/'run_metadata.json')
    if a['region_word_initialization']!=b['region_word_initialization']:
        raise ValueError('Fine/control initial region-token projection fingerprints differ')
    if a['optimizer_initialization']!=b['optimizer_initialization']:
        raise ValueError('Fine/control optimizer groups, parameter counts or LRs differ')
    a=r.base.read_json(left/'data_split_manifest.json');b=r.base.read_json(right/'data_split_manifest.json')
    if any(a[k]!=b[k] for k in ('indices','dataset_ids_sha256','test_ids_sha256')):
        raise ValueError('Fine/control actual split memberships differ')


def make_runner():
    r=previous.make_runner(variants=VARIANTS,version=VERSION,
        protocol_doc='docs/align_fixed_temperature_fine.md',queues={'low':('tau005',),'high':('tau020',)})
    r.__doc__=__doc__
    def control(ds,split,source):
        old=previous_runner(r.ROOT)
        old.verify('tau010',ds,split,source)
        return old.paths('tau010',ds,split)
    r.control=control
    original=r.verify
    def verify(v,ds,split,source):
        original(v,ds,split,source)
        control_dir,_=control(ds,split,source)
        compare_common_initialization_and_split(r,r.paths(v,ds,split)[0],control_dir)
    r.verify=verify
    def extra(source):
        controls={}
        for ds in r.DATASETS:
            d,t=control(ds,None,source)
            controls[ds]={'run_id':t.stem,'csv_sha256':r.base.digest(t),
                          'completion_sha256':r.base.digest(d/'completion_manifest.json')}
        return {'fixed_temperature_fine_protocol.json':{
            'version':VERSION,'source':source,'new_fixed_temperatures':VARIANTS,
            'reference_temperature':.1,'reference_version':previous.VERSION,
            'reference_controls':controls,'reference_retrained':False,
            'new_runs':4,'epochs_each':60,'nss':'original legacy six-dimensional',
            'pooling':'region/BPE-token mean; same lexical mask; no coverage or native-score blend',
            'selection':'Quality SROCC and arithmetic-mean joint, each validation-selected'}}
    r.pack_extra_records=extra
    return r

if __name__=='__main__':make_runner().main()
