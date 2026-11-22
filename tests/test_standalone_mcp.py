import asyncio,json,hashlib
from pathlib import Path
import numpy as np
import pytest
from weight_space_lab.standalone import Workspace,helpers

def reference(p):return {'path':str(p.name if False else 'exports/'+p.name),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def test_hash_bound_algebra_and_explicit_edit_values(tmp_path):
 d=tmp_path/'exports';d.mkdir();p=d/'w.npy';np.save(p,np.array([[1.,2.],[3.,4.]]))
 w=Workspace(tmp_path);app=helpers(w)
 async def call(name,args):
  r=await app.call_tool(name,args)
  if hasattr(r,'content'):return getattr(r,'structured_content',None) or json.loads(r.content[0].text)
  return json.loads(r[0].text)
 r=asyncio.run(call('array_compute',{'operation':'mean','inputs':[reference(p)],'output':'analysis/mean.npy','options':{'axis':0}}))
 assert np.array_equal(w.ref({'path':r['path'],'sha256':r['sha256']}),[2.,3.])
 assert w.expand({'direction':{'$array':{'path':r['path'],'sha256':r['sha256']}}})=={'direction':[2.,3.]}
 # A stale hash and a path outside the workspace must never silently alter numeric plans.
 np.save(p,np.zeros((2,2)))
 with pytest.raises(ValueError,match='hash mismatch'):w.ref({'path':r['path'],'sha256':'0'*64})
 with pytest.raises(ValueError):w.ref({'path':'../../external.npy','sha256':'0'*64})
 with pytest.raises(ValueError):w.expand({'$array':{'path':'model/config.json','sha256':'0'*64}})
 # Division by zero is an error and must not produce a usable array.
 with pytest.raises(Exception):asyncio.run(call('array_compute',{'operation':'divide','inputs':[reference(p),reference(p)],'output':'analysis/invalid.npy'}))
 assert not (tmp_path/'analysis/invalid.npy').exists()

def test_frozen_limits_and_protected_parameter_paths(tmp_path):
 (tmp_path/'protocol.json').write_text(json.dumps({'limits':{'max_relative_l2':.15,'max_changed_fraction':.002}}))
 w=Workspace(tmp_path);p={'checkpoint':'model/model.safetensors','limits':{'max_relative_l2':.15,'max_changed_fraction':.002},'operations':[{'tensor':'model.layers.0.mlp.down_proj.weight','kind':'scale'}]}
 assert w.check_plan(p)==p
 with pytest.raises(ValueError):w.check_plan({**p,'limits':{**p['limits'],'max_changed_fraction':.01}})
 with pytest.raises(ValueError):w.check_plan({**p,'operations':[{'tensor':'lm_head.weight','kind':'scale'}]})
 with pytest.raises(ValueError):w.new('../outside.json','analysis')

def test_missing_operation_fields_give_actionable_errors(tmp_path):
 """A missing 'axis' or 'slices' used to surface as a bare KeyError; the message now names the required field."""
 from weight_space_lab.core import _transform
 with pytest.raises(ValueError,match='axis must be 0'):_transform(np.zeros((2,2)),{'kind':'scale'})
 d=tmp_path/'exports';d.mkdir();p=d/'w.npy';np.save(p,np.zeros((3,3)))
 app=helpers(Workspace(tmp_path))
 with pytest.raises(Exception,match='slice requires "slices"'):asyncio.run(app.call_tool('array_compute',{'operation':'slice','inputs':[reference(p)],'output':'analysis/s.npy'}))
