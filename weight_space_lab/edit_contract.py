"""Public edit-plan schemas grounded in core._apply_operation/_transform."""
from copy import deepcopy


def plan_schema(array_refs=False):
    indices={'type':'array','items':{'type':'integer','minimum':0},'minItems':1,'uniqueItems':True}
    vector={'type':'array','items':{'type':'number'},'minItems':1}
    matrix={'type':'array','items':vector,'minItems':1}
    ref={'type':'object','properties':{'$array':{'type':'object','properties':{'path':{'type':'string'},'sha256':{'type':'string','pattern':'^[0-9a-f]{64}$'}},'required':['path','sha256'],'additionalProperties':False}},'required':['$array'],'additionalProperties':False}
    v={'anyOf':[vector,ref]} if array_refs else vector
    m={'anyOf':[matrix,ref]} if array_refs else matrix
    op={'type':'object','properties':{'kind':{'enum':['set_values','scale','project_out','rank_one']},'tensor':{'type':'string','minLength':1},'axis':{'type':'integer','enum':[0,1],'description':'Select stored rows (0) or columns (1); inspect actual module layout first.'},'indices':indices,'strength':{'type':'number','minimum':0,'description':'scale: Wblock_new=(1-strength)*Wblock; 0 is identity, 1 ablates. project_out/rank_one use the explicit nonnegative alpha, with no upper bound except declared numeric budgets.'},'direction':v,'left':v,'right':v,'rows':indices,'columns':indices,'expected_values':m,'values':m},'required':['kind','tensor'],'additionalProperties':False,'allOf':[]}
    for kind,required in [('set_values',['rows','columns','expected_values','values']),('scale',['axis','indices','strength']),('project_out',['axis','indices','strength','direction']),('rank_one',['axis','indices','strength','left','right'])]:
        rule={'if':{'properties':{'kind':{'const':kind}},'required':['kind']},'then':{'required':required}}
        if kind=='scale':rule['then']['properties']={'strength':{'maximum':1}}
        op['allOf'].append(rule)
    sequence={'type':'object','properties':{'kind':{'const':'sequence'},'tensor':{'type':'string'},'steps':{'type':'array','items':op,'minItems':1,'maxItems':32}},'required':['kind','tensor','steps'],'additionalProperties':False}
    joint=deepcopy(op);joint['properties']['kind']['enum'].append('sequence');joint['properties']['steps']=sequence['properties']['steps'];joint['allOf'].append({'if':{'properties':{'kind':{'const':'sequence'}},'required':['kind']},'then':{'required':['steps']}})
    return {'type':'object','properties':{'checkpoint':{'type':'string','description':'Workspace-relative original model/model.safetensors; this is a file, not the model directory.'},'source_sha256':{'type':'string','pattern':'^[0-9a-f]{64}$','description':'Actual original safetensors hash returned by inspect_checkpoint/experiment_info.'},'hypothesis':{'type':'string','minLength':1},'limits':{'type':'object','properties':{'max_changed_fraction':{'type':'number','exclusiveMinimum':0,'maximum':1},'max_relative_l2':{'type':'number','exclusiveMinimum':0,'maximum':1}},'required':['max_changed_fraction','max_relative_l2'],'additionalProperties':False},'operations':{'type':'array','items':joint,'minItems':1,'maxItems':32}},'required':['checkpoint','source_sha256','hypothesis','limits','operations'],'additionalProperties':False}


def edit_api_schema():
    return {'plan_schema':plan_schema(False),'array_reference_plan_schema':plan_schema(True),'semantics':{
        'set_values':'Read the current selected block and require expected_values to match it exactly after stored-dtype rounding; values specifies replacement rows x columns, at most4096 entries.',
        'scale':'For selected stored rows/columns, multiply by (1-strength). strength=0 is identity; strength=1 deletes the selected block. It is not a direct multiplier.',
        'project_out':'Normalize direction to u along the opposite stored axis. axis0: B-alpha*(B@u)u.T. axis1: B-alpha*u(u.T@B). Full opposite-axis width is required.',
        'rank_one':'B_new=B+strength*outer(left,right). left length is selected block rows; right length is selected block columns. Vector signs may be negative; strength must be nonnegative.',
        'sequence':'Apply steps in order on the same tensor with stored-dtype rounding after each step. Only one top-level operation per tensor.',
        'arrays':'Only *_from_arrays tools expand {$array:{path,sha256}} values. preview_edit requires actual numeric vectors; preview_edit_from_arrays handles references.',
        'revision':'create_candidate starts from original. revise_candidate or create_candidate_from_arrays(parent_candidate_dir=...) appends only this revision operations to the real parent. rationale requires observation,hypothesis,value_derivation,prediction,disconfirming_result.',
        'readiness':'Before semantic edits, prepare source-bound evidence and inspect decision/reasons. Ready is only readiness for a causal probe, not a causal/performance result.'},'selection_policy':'Caller chooses tensor, explicit coordinates, directions, strengths and values. Schema discovery performs no fitting, forward or edits. No successful coordinates/coefficients are supplied.'}


def enrich_catalog(catalog,sdk_major):
    for name,tool in catalog.items():
        if name not in ('preview_edit','preview_edit_from_arrays','create_candidate','create_candidate_from_arrays','research_prepare_edit','prepare_edit_from_arrays'):continue
        key='input_schema' if sdk_major>=2 else 'inputSchema'
        schema=deepcopy(getattr(tool,key));schema['properties']['plan']=plan_schema(name.endswith('_from_arrays'))
        setattr(tool,key,schema)
    return catalog


def error_detail(error):
    """SDK2 wraps domain failures; return their cause without a full private traceback."""
    import jsonschema
    seen=set();cause=error
    while cause.__cause__ is not None and id(cause) not in seen:
        seen.add(id(cause));cause=cause.__cause__
    if isinstance(cause,jsonschema.ValidationError):return {'error_type':'InvalidToolArguments','message':cause.message,'argument_path':list(cause.absolute_path)}
    if isinstance(cause,KeyError):return {'error_type':'MissingField','message':'Missing required field: '+str(cause.args[0])}
    return {'error_type':type(cause).__name__,'message':str(cause) or type(cause).__name__}
