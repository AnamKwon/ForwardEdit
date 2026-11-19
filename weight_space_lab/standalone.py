"""User-facing, MCP-only inference/edit workflow; no loop service or code executor.

python -m weight_space_lab.standalone --workspace /absolute/workspace
"""
from __future__ import annotations
import argparse,asyncio,contextlib,hashlib,json,math,os,time,uuid
from pathlib import Path
import numpy as np
from mcp.server import Server
from .mcp_compat import FastMCP, SDK_MAJOR, schema
from mcp.server.stdio import stdio_server
from mcp import types
from .server import build_server
from . import core,research
from .edit_contract import edit_api_schema as contract_schema, enrich_catalog, error_detail

PROJECT=Path(__file__).resolve().parents[1]
CORE_TOOLS={'inspect_checkpoint','read_tensor_slice','describe_parameter','describe_model_structure','inspect_module_computation','read_parameter_entries','analyze_matrix_geometry','research_forward','research_status','research_result','research_read_generations','research_read_array','research_make_contexts','research_export_tensor','research_prepare_edit','create_candidate','revise_candidate','compare_candidate_parameters','preview_edit'}

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump_new(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())

class Workspace:
 def __init__(self,root):self.root=Path(root).resolve()
 def path(self,s):return core.local(self.root,s)
 def new(self,s,prefix):
  p=self.path(s)
  if not p.is_relative_to(self.root/prefix) or p.exists():raise ValueError('new path under '+prefix+' required')
  return p
 def ref(self,ref):
  if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('array reference requires path and sha256')
  p=self.path(ref['path'])
  if p.suffix!='.npy' or not any(p.is_relative_to(self.root/k) for k in ('analysis','research_jobs','exports')):raise ValueError('array must be a workspace analysis/export/observed artifact')
  before=digest(p)
  if before!=ref['sha256']:raise ValueError('array hash mismatch')
  a=np.load(p,allow_pickle=False)
  if a.dtype.kind not in 'fiu' or a.size>16_777_216 or not np.isfinite(a).all():raise ValueError('bounded finite numeric array required')
  if digest(p)!=before:raise ValueError('array changed during read')
  return np.asarray(a,dtype=np.float64)
 def expand(self,value):
  if isinstance(value,dict):
   if set(value)=={'$array'}:return self.ref(value['$array']).tolist()
   return {k:self.expand(v) for k,v in value.items()}
  if isinstance(value,list):return [self.expand(v) for v in value]
  return value
 def note(self,kind,data):
  p=self.new(f'notes/{kind}-{uuid.uuid4().hex}.json','notes');dump_new(p,data)
  return {'path':str(p.relative_to(self.root)),'sha256':digest(p)}
 def check_plan(self,plan):
  if self.path(plan['checkpoint'])!=self.root/'model/model.safetensors':raise ValueError('original model checkpoint required')
  limits=plan['limits']
  protocol=self.root/'protocol.json'
  if protocol.exists():
   cap=json.loads(protocol.read_text()).get('limits',{})
   for key,maximum in cap.items():
    if limits.get(key,math.inf)>maximum:raise ValueError('edit exceeds declared '+key)
  def check(op):
   name=op['tensor']
   if any(t in name for t in ('embed_tokens','lm_head')) or name=='model.norm.weight':raise ValueError('embedding/output head/final norm edit excluded from this research interface')
   if op.get('kind')=='sequence':
    for x in op['steps']:check({**x,'tensor':name})
  for op in plan['operations']:check(op)
  return plan

def helpers(w):
 app=FastMCP('forwardedit-analysis')
 @app.tool()
 def edit_api_schema()->dict:
  """Read exact required edit-plan/operation fields, axis and strength semantics before authoring a plan. No coordinates or coefficients are chosen."""
  return contract_schema()
 @app.tool()
 def workspace_status()->dict:
  """Discover existing current-study inference jobs/candidates and cumulative budgets for an MCP-only resume. Never launches inference or selects edits."""
  c=research.config(w.root)
  jobs=[]
  for p in (w.root/'research_jobs').glob('*/snapshot.json'):
   snap=json.loads(p.read_text());state=json.loads((p.parent/'state.json').read_text());spec=snap['spec']
   jobs.append({'job_id':p.parent.name,'status':state['status'],'type':spec['type'],'model':spec['model'],'split':spec['split'],'items':len(spec['ids']),'snapshot_path':str(p.relative_to(w.root))})
  return {'successful_completed':sum(j['status']=='succeeded' for j in jobs),'accepted_attempts':len(jobs),'successful_budget':c['successful_forward_budget'],'attempt_budget':c['attempt_budget'],'jobs':jobs[-50:],'candidates':[str(p.parent.relative_to(w.root)) for p in (w.root/'candidates').glob('*/manifest.json')],'artifact_listing_tool':'list_workspace_artifacts','protocol_path':'protocol.json'}
 @app.tool()
 def list_workspace_artifacts(prefix:str='analysis',offset:int=0,limit:int=40)->dict:
  """List bounded current-workspace analysis/export/note/job/candidate artifact paths for independent discovery and resume."""
  if prefix not in ('analysis','exports','notes','research_jobs','candidates') or not 0<=offset or not 1<=limit<=100:raise ValueError('select a generated artifact prefix and bounded page')
  paths=sorted(p for p in (w.root/prefix).rglob('*') if p.is_file())
  selected=[{'path':str(w.path(str(p)).relative_to(w.root)),'bytes':p.stat().st_size} for p in paths[offset:offset+limit]]
  return {'total':len(paths),'offset':offset,'files':selected,'next_offset':offset+limit if offset+limit<len(paths) else None}
 @app.tool()
 def preview_edit_from_arrays(plan:dict)->dict:
  """Validate/numerically preview the caller's exact edit, expanding hash-bound numeric refs; no candidate or forward."""
  return core.prepare(w.root,w.check_plan(w.expand(plan)))[1]
 @app.tool()
 def experiment_info()->dict:
  """Read this workspace's immutable protocol, model/data paths and development counts; no test access."""
  c=research.config(w.root);data=research.dataset(w.root,c['dataset_path'])
  p=w.root/'protocol.json'
  return {'protocol':json.loads(p.read_text()) if p.exists() else {},'research_config':c,'development_counts':{k:len(v) for k,v in data.items()},'guidance_entry':'skills/weight-space-engineer/SKILL.md','only_model_execution_tools':['research_forward','forward_and_wait'],'array_operations':'mean,sum,add,subtract,multiply,divide,scale,normalize,norm,dot,matmul,transpose,slice,take,concatenate,reshape,abs,topk; no fitting/optimizer/execution'}
 @app.tool()
 def read_guidance(path:str='skills/weight-space-engineer/SKILL.md')->dict:
  """Read shipped project skills or inference/parameter-analysis contracts. No historical experiment recipes."""
  p=(PROJECT/path).resolve()
  if not (p.is_relative_to(PROJECT/'skills') or p in [PROJECT/'docs/INFERENCE_ONLY_CONTRACT.md',PROJECT/'docs/PARAMETER_ANALYSIS.md']) or p.suffix!='.md':raise ValueError('project skill or contract required')
  return {'path':path,'sha256':digest(p),'text':p.read_text()[:30000]}
 @app.tool()
 def development_examples(split:str='discovery',ids:list[str]|None=None,offset:int=0,limit:int=8)->dict:
  """Read exact development prompts, gold answers and reference solutions; no withheld test."""
  if split not in ('discovery','validation') or not 0<=offset or not 1<=limit<=64:raise ValueError('invalid development selection')
  rows=research.dataset(w.root,research.config(w.root)['dataset_path']).get(split,[])
  if ids is not None:
   if len(ids)>64 or len(ids)!=len(set(ids)):raise ValueError('select at most64 unique IDs')
   by={r['id']:r for r in rows}
   if not set(ids)<=by.keys():raise ValueError('unknown development ID')
   selected=[by[i] for i in ids]
  else:selected=rows[offset:offset+limit]
  return {'split':split,'total':len(rows),'rows':selected}
 @app.tool()
 def read_artifact(path:str,offset:int=0,max_chars:int=12000)->dict:
  """Read bounded text from current workspace generated artifacts, model config or protocol; never executes files."""
  p=w.path(path)
  allowed=any(p.is_relative_to(w.root/k) for k in ('analysis','exports','research_jobs','candidates','notes','model')) or p in (w.root/'protocol.json',w.root/'research.json')
  if not allowed or p.suffix not in ('.json','.jsonl','.txt','.log','.jinja') or not 0<=offset or not 1<=max_chars<=16000:raise ValueError('bounded current-workspace text artifact required')
  if p.stat().st_size>100_000_000:raise ValueError('text artifact too large')
  text=p.read_text()
  return {'path':path,'sha256':digest(p),'characters':len(text),'offset':offset,'text':text[offset:offset+max_chars]}
 @app.tool()
 def array_compute(operation:str,inputs:list[dict],output:str,options:dict|None=None)->dict:
  """Execute caller-selected finite CPU array algebra on hash-bound exports/actual observations. Does not choose weights or fit loss. See MCP-only skill reference for exact options."""
  o=options or {};a=[w.ref(r) for r in inputs];axis=o.get('axis');axis=tuple(axis) if isinstance(axis,list) else axis;keep=o.get('keepdims',False)
  if not 1<=len(a)<=64:raise ValueError('1..64 bound inputs required')
  if operation in ('add','subtract','multiply','divide','dot','matmul'):
   if len(a)!=2:raise ValueError('two inputs required')
   with np.errstate(all='raise'):z=getattr(np,operation)(*a)
  elif operation=='mean':z=np.mean(a[0],axis=axis,keepdims=keep)
  elif operation=='sum':z=np.sum(a[0],axis=axis,keepdims=keep)
  elif operation=='scale':z=a[0]*float(o['factor'])
  elif operation=='norm':z=np.linalg.norm(a[0],axis=axis,keepdims=keep)
  elif operation=='normalize':
   n=np.linalg.norm(a[0],axis=axis,keepdims=True)
   if np.any(n==0):raise ValueError('zero norm')
   z=a[0]/n
  elif operation=='transpose':z=np.transpose(a[0],axes=o.get('axes'))
  elif operation=='slice':
   ix=[]
   for v in o['slices']:
    if type(v) is int:ix.append(v)
    elif isinstance(v,list) and len(v) in (2,3):ix.append(slice(*v))
    else:raise ValueError('slices entries are integers or [start,stop,step]')
   z=a[0][tuple(ix)]
  elif operation=='take':z=np.take(a[0],o['indices'],axis=axis)
  elif operation=='concatenate':z=np.concatenate(a,axis=axis if axis is not None else 0)
  elif operation=='reshape':z=a[0].reshape(o['shape'])
  elif operation=='abs':z=np.abs(a[0])
  elif operation=='topk':
   if a[0].ndim!=1:raise ValueError('topk requires vector')
   k=o['k']
   if type(k) is not int or not 1<=k<=min(256,a[0].size):raise ValueError('bounded topk required')
   score=np.abs(a[0]) if o.get('absolute',False) else a[0]
   ix=np.argsort(-score,kind='stable')[:k];z=ix
   extra={'top_values':a[0][ix].tolist()}
  else:raise ValueError('unsupported CPU algebra operation')
  z=np.asarray(z)
  if z.size>16_777_216 or z.dtype.kind not in 'fiu' or not np.isfinite(z).all():raise ValueError('result must be bounded finite numeric array')
  p=w.new(output,'analysis');p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:np.save(f,z,allow_pickle=False)
  result={'path':str(p.relative_to(w.root)),'sha256':digest(p),'shape':list(z.shape),'dtype':str(z.dtype),'norm':float(np.linalg.norm(z)),'min':float(np.min(z)) if z.size else None,'max':float(np.max(z)) if z.size else None,'preview':z.reshape(-1)[:16].tolist(),'model_forward_calls':0,'operation':operation,'inputs':inputs,'options':o}
  if operation=='topk':result.update(extra)
  dump_new(Path(str(p)+'.json'),result)
  return result
 @app.tool()
 def record_observation(observation:dict)->dict:
  """Retain caller-authored hypothesis, numerical rationale, feedback or decision. No weight selection or evaluation."""
  return w.note('observation',observation)
 @app.tool()
 def prepare_edit_from_arrays(plan:dict,tensor:str,rows:list[int],columns:list[int],contexts:list[dict],direction:dict,basis:dict,output:str)->dict:
  """Run actual research_prepare_edit after expanding explicit {$array:{path,sha256}} numeric plan references; direction is an array reference. No fitting."""
  from .research_analysis import research_prepare_edit
  p=w.check_plan(w.expand(plan));w.new(output,'analysis')
  w.note('resolved-proposal',{'plan_template':plan,'resolved_plan':p,'direction':direction,'basis':basis})
  return research_prepare_edit(w.root,p,tensor,rows,columns,contexts,w.ref(direction).tolist(),basis,output)
 @app.tool()
 def create_candidate_from_arrays(plan:dict,output:str,parent_candidate_dir:str|None=None,rationale:dict|None=None)->dict:
  """Write the caller's explicit numeric plan (optionally using hash-bound numeric refs). With parent, plan.operations are this revision only and rationale must describe observed feedback. No generated coordinates or strengths."""
  p=w.check_plan(w.expand(plan));w.new(output,'candidates')
  protocol=w.root/'protocol.json'
  cap=json.loads(protocol.read_text()).get('max_candidates',32) if protocol.exists() else 32
  if len(list((w.root/'candidates').glob('*/manifest.json')))>=cap:raise ValueError('candidate budget exhausted')
  w.note('resolved-edit',{'plan_template':plan,'resolved_plan':p,'parent_candidate_dir':parent_candidate_dir,'rationale':rationale,'output':output})
  if parent_candidate_dir:
   from .lineage import revise_candidate
   return revise_candidate(w.root,'model',parent_candidate_dir,p['operations'],p['hypothesis'],rationale or {},p['limits'],output)
  return core.write_candidate(w.root,p,output)
 @app.tool()
 async def forward_and_wait(spec:dict)->dict:
  """Run actual project research_forward and await its verified terminal result without consuming editor turns on polling. One logical successful batch; never auto-retries. Full job ID saved before waiting."""
  result=await asyncio.to_thread(research.research_forward,w.root,spec);w.note('submitted-forward',{'spec':spec,'submission':result});jid=result['job_id']
  while True:
   state=await asyncio.to_thread(research.research_status,w.root,jid)
   if state['status'] in ('succeeded','failed'):
    return await asyncio.to_thread(research.research_result,w.root,jid)
   await asyncio.sleep(10)
 @app.tool()
 def generation_window(job_id:str,identifier:str,which:str='original',offset:int=0,max_chars:int=6000,from_end:bool=False)->dict:
  """Read a bounded window from a real saved generation, including late reasoning or final answer; no inference."""
  if which not in ('original','edited') or not 0<=offset or not 1<=max_chars<=10000:raise ValueError('invalid window')
  result=research.research_result(w.root,job_id)
  artifact=next(a for a in result['artifacts'] if a.get('kind')=='generations' and a.get('model')==which)
  rows=json.loads(w.path(artifact['path']).read_text());row=next(r for r in rows if r['id']==identifier)
  text=row['text'];start=max(0,len(text)-offset-max_chars) if from_end else offset
  return {'id':identifier,'job_id':job_id,'which':which,'artifact_sha256':artifact['sha256'],'grade':row['grade'],'generated_tokens':row['generated_tokens'],'finish_reason':row['finish_reason'],'extracted_answer':row.get('extracted_answer'),'characters':len(text),'start':start,'text':text[start:start+max_chars]}
 @app.tool()
 def freeze_candidate(candidate_dir:str,evidence_job_ids:list[str],decision:str,explanation:str)->dict:
  """Freeze one caller-selected candidate and its evidence once. This records selection, never certifies a performance gain. Held-out evaluation is separate."""
  from .evaluation import verify_candidate
  verify_candidate(w.root,'model',candidate_dir)
  jobs=[research.research_result(w.root,j) for j in evidence_job_ids]
  d=w.path(candidate_dir);record={'candidate_dir':candidate_dir,'manifest_sha256':digest(d/'manifest.json'),'patch_sha256':digest(d/'replacements.safetensors'),'evidence_job_ids':evidence_job_ids,'decision':decision,'explanation':explanation,'utc':time.time(),'performance_certified':False}
  dump_new(w.root/'frozen-candidate.json',record);return record
 return app

async def serve(root):
 w=Workspace(root);base=build_server(w.root);extra=helpers(w)
 catalog={t.name:t for t in await base.list_tools() if t.name in CORE_TOOLS};catalog.update({t.name:t for t in await extra.list_tools()});enrich_catalog(catalog,SDK_MAJOR)
 audit=w.root/'mcp-audit';audit.mkdir(exist_ok=True)
 async def call_tool(name,arguments):
  token=uuid.uuid4().hex;start=time.monotonic();dump_new(audit/(token+'-request.json'),{'name':name,'arguments':arguments,'utc':time.time()})
  try:
   import jsonschema
   if name not in catalog:raise ValueError('unavailable tool')
   jsonschema.validate(arguments,schema(catalog[name]))
   if name in ('create_candidate','preview_edit','research_prepare_edit'):w.check_plan(arguments['plan'])
   if name=='research_export_tensor':w.new(arguments['output'],'exports')
   if name in ('research_make_contexts','research_prepare_edit'):w.new(arguments['output'],'analysis')
   if name=='revise_candidate':w.check_plan({'checkpoint':'model/model.safetensors','limits':arguments['limits'],'operations':arguments['operations']})
   if name in ('create_candidate','revise_candidate'):
    protocol=w.root/'protocol.json';cap=json.loads(protocol.read_text()).get('max_candidates',32) if protocol.exists() else 32
    if len(list((w.root/'candidates').glob('*/manifest.json')))>=cap:raise ValueError('candidate budget exhausted')
    w.new(arguments['output'],'candidates')
    if (w.root/'frozen-candidate.json').exists():raise ValueError('candidate already frozen')
   if name=='create_candidate_from_arrays' and (w.root/'frozen-candidate.json').exists():raise ValueError('candidate already frozen')
   # FastMCP retains each original project implementation and its typed input validation.
   result=await (base if name in CORE_TOOLS else extra).call_tool(name,arguments)
   if isinstance(result,types.CallToolResult):
    content=result.content;structured=getattr(result,'structured_content',None) if SDK_MAJOR>=2 else result.structuredContent
    failed=result.is_error if SDK_MAJOR>=2 else result.isError
    if failed:raise ValueError('Project tool returned an error: '+str(content))
   elif isinstance(result,tuple):content,structured=result
   elif isinstance(result,dict):content=[types.TextContent(type='text',text=json.dumps(result))];structured=result
   else:content=list(result);structured=None
   value={'content':[c.model_dump(mode='json',by_alias=True) for c in content],'structuredContent':structured}
   dump_new(audit/(token+'-response.json'),{'name':name,'ok':True,'elapsed_seconds':time.monotonic()-start,'utc':time.time(),'result':value})
   return types.CallToolResult(content=content,**({'structured_content':structured,'is_error':False} if SDK_MAJOR>=2 else {'structuredContent':structured,'isError':False}))
  except Exception as e:
   detail=error_detail(e)
   dump_new(audit/(token+'-response.json'),{'name':name,'ok':False,'elapsed_seconds':time.monotonic()-start,'utc':time.time(),'error':detail['message'],'error_detail':detail})
   return types.CallToolResult(content=[types.TextContent(type='text',text=json.dumps(detail,ensure_ascii=False))],**({'is_error':True} if SDK_MAJOR>=2 else {'isError':True}))
 if SDK_MAJOR>=2:
  async def list_handler(ctx,params):return types.ListToolsResult(tools=list(catalog.values()))
  async def call_handler(ctx,params):return await call_tool(params.name,params.arguments or {})
  server=Server('forwardedit',on_list_tools=list_handler,on_call_tool=call_handler,get_tool_input_schema=lambda n:schema(catalog[n]) if n in catalog else None)
 else:
  server=Server('forwardedit')
  @server.list_tools()
  async def list_tools():return list(catalog.values())
  server.call_tool(validate_input=False)(call_tool)
 async with stdio_server() as (rx,tx):await server.run(rx,tx,server.create_initialization_options())

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',required=True);args=parser.parse_args()
 from .inference_guards import install_inference_guards
 install_inference_guards()
 # The wakeup also remains active through executor shutdown in restricted SDK environments.
 loop=asyncio.new_event_loop();asyncio.set_event_loop(loop)
 async def pulse():
  while True:await asyncio.sleep(.02)
 beat=loop.create_task(pulse())
 try:loop.run_until_complete(serve(Path(args.workspace).resolve()))
 finally:
  tasks=[t for t in asyncio.all_tasks(loop) if t is not beat]
  for t in tasks:t.cancel()
  if tasks:loop.run_until_complete(asyncio.gather(*tasks,return_exceptions=True))
  loop.run_until_complete(loop.shutdown_asyncgens());loop.run_until_complete(loop.shutdown_default_executor());beat.cancel()
  with contextlib.suppress(asyncio.CancelledError):loop.run_until_complete(beat)
  asyncio.set_event_loop(None);loop.close()
if __name__=='__main__':main()
