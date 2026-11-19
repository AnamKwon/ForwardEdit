import jsonschema
import numpy as np
from safetensors.numpy import save_file
from weight_space_lab.core import prepare,sha256
from weight_space_lab.edit_contract import plan_schema,error_detail


def test_published_schema_and_core_agree_on_scale_and_required_fields(tmp_path):
    p=tmp_path/'model.safetensors';save_file({'linear.weight':np.eye(2,dtype=np.float32)},str(p))
    plan={'checkpoint':'model.safetensors','source_sha256':sha256(p),'hypothesis':'contract control','limits':{'max_changed_fraction':.5,'max_relative_l2':1},'operations':[{'kind':'scale','tensor':'linear.weight','axis':0,'indices':[0],'strength':1}]}
    jsonschema.validate(plan,plan_schema())
    tensors,_=prepare(tmp_path,plan)
    assert np.array_equal(tensors['linear.weight'],[[0,0],[0,1]])
    plan['operations'][0].pop('strength')
    try:jsonschema.validate(plan,plan_schema())
    except jsonschema.ValidationError as exc:assert 'strength' in error_detail(exc)['message']
    else:raise AssertionError('Missing coefficient accepted by published contract')


def test_sdk_wrapped_domain_error_is_reported_without_losing_cause():
    try:
        try:raise ValueError('direction must match opposite axis')
        except ValueError as original:raise RuntimeError('Error executing tool preview_edit') from original
    except RuntimeError as wrapped:
        detail=error_detail(wrapped)
    assert detail=={'error_type':'ValueError','message':'direction must match opposite axis'}
