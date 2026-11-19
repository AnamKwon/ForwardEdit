import json
import numpy as np
import pytest
from safetensors.numpy import save_file
from weight_space_lab import research_analysis as a, core


def world(root,monkeypatch):
    (root/'model').mkdir();(root/'model/config.json').write_text('{}')
    save_file({'writer.weight':np.tile(np.array([[.4,.2]],dtype=np.float32),(80,1))},root/'model/model.safetensors')
    artifacts=[]
    for name,x in [('normal',[1.,0.]),('loop',[-2.,0.])]:
        np.save(root/(name+'.npy'),np.array([x],dtype=np.float32))
        artifacts.append({'path':name+'.npy','kind':'observations','model':'original','module':'writer','id':name,'sha256':core.sha256(root/(name+'.npy')),'positions':[0]})
    sha=core.sha256(root/'model/model.safetensors')
    monkeypatch.setattr(a,'_job',lambda *args:({'provenance':{'model/model.safetensors':sha}}, {'config':{'model_dir':'model','inference_dtype':'float32'},'spec':{'model':'original','type':'generate','split':'discovery'}},artifacts))
    plan={'checkpoint':'model/model.safetensors','source_sha256':sha,'hypothesis':'Known fixture: explicit selected writer attenuation','limits':{'max_changed_fraction':.6,'max_relative_l2':.6},'operations':[{'tensor':'writer.weight','kind':'scale','axis':1,'indices':[0],'strength':.5}]}
    contexts=[{'role':'reference','job_id':'fixture','path':'normal.npy'},{'role':'failure','job_id':'fixture','path':'loop.npy'}]
    basis={k:'Explicit fixture hypothesis only' for k in ['failure_description','structural_role','mechanism_hypothesis','pairing_justification','value_derivation','alternative_explanation','expected_behavior','disconfirming_result']}
    basis.update(controls=[{'kind':'opposing','purpose':'challenge signed prediction'}],max_reference_output_delta=2.)
    return dict(plan=plan,tensor='writer.weight',rows=list(range(80)),columns=[0],contexts=contexts,direction=[1.]*80,basis=basis,output='evidence.json')


def test_wide_actual_artifact_affine_evidence_no_update(tmp_path,monkeypatch):
    args=world(tmp_path,monkeypatch);before=core.sha256(tmp_path/'model/model.safetensors')
    result=a.research_prepare_edit(tmp_path,**args)
    assert result['decision']=='ready_for_causal_probe'
    assert result['distinct_problem_ids']==2
    np.testing.assert_allclose(result['max_reference_delta_l2'],np.sqrt(80)*.2,atol=1e-7)
    np.testing.assert_allclose(result['mean_failure_delta_projection'],np.sqrt(80)*.4,atol=1e-7)
    assert result['model_forward_calls']==0 and not result['candidate_created']
    assert before==core.sha256(tmp_path/'model/model.safetensors')
    args['basis']={};args['output']='defer.json'
    assert a.research_prepare_edit(tmp_path,**args)['decision']=='defer'


def test_tampered_wrong_owner_duplicate_vectors_rejected(tmp_path,monkeypatch):
    args=world(tmp_path,monkeypatch)
    args['contexts']*=2
    with pytest.raises(ValueError,match='duplicate context'):a.research_prepare_edit(tmp_path,**args)
    args['contexts']=args['contexts'][:2]
    np.save(tmp_path/'normal.npy',np.zeros((1,2),dtype=np.float32))
    with pytest.raises(ValueError,match='hash mismatch'):a.research_prepare_edit(tmp_path,**args)


def test_uncovered_weight_changes_defer(tmp_path,monkeypatch):
    args=world(tmp_path,monkeypatch);args['rows']=[0];args['direction']=[1.]
    result=a.research_prepare_edit(tmp_path,**args)
    assert result['decision']=='defer'
    assert 'omits incremental' in ' '.join(result['reasons'])


def test_parameter_export_is_exact_nonmutating_and_bound(tmp_path):
    source=np.arange(60,dtype=np.float32).reshape(3,4,5)
    save_file({'cube':source},tmp_path/'source.safetensors')
    result=a.research_export_tensor(tmp_path,'source.safetensors','cube','arrays/cube.npy')
    np.testing.assert_array_equal(np.load(tmp_path/result['array_path'],allow_pickle=False),source)
    assert result['array_sha256']==core.sha256(tmp_path/'arrays/cube.npy')
    assert result['source_sha256']==core.sha256(tmp_path/'source.safetensors')
    assert result['model_forward_calls']==0
    with pytest.raises(ValueError,match='inside workspace'):
        a.research_export_tensor(tmp_path,'source.safetensors','cube','../outside.npy')
