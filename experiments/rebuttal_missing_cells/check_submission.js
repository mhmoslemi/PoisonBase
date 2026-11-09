// Mock the complete batch; no Slurm submission, Python, or training.
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert'),crypto=require('crypto');
const {spawnSync}=require('child_process'),root=path.resolve(__dirname,'../..');
const name='sbatch/rebuttal_missing_cells_vulcan',p=JSON.parse(fs.readFileSync(path.join(root,name,'plan.json')));
assert.equal(p.cells.length,10);assert.equal(p.tasks.length,52);
const source=JSON.parse(fs.readFileSync(path.join(root,p.source_plan)));
const overrideFile=path.join(root,name,'account_overrides.tsv');
const overrides=new Map(fs.existsSync(overrideFile)?fs.readFileSync(overrideFile,'utf8').trim().split('\n').filter(Boolean).map(l=>l.split('\t')):[]);
for(const c of p.cells){const original=source.cells.find(s=>s.id===c.source_cell);assert.deepEqual(c.targets,original.targets);assert.equal(c.targets.length,4);assert.equal(c.args.num_victims,4);assert.equal(c.args.victim_epochs,50);const expected={...original.args,base:c.component==='random'?'random':'ours',lambda_margin:1};assert.deepEqual(c.args,expected);}
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'missing-cells-')),batch=path.join(tmp,name),project=path.join(tmp,'project');
try{
    fs.mkdirSync(path.dirname(batch),{recursive:true});fs.cpSync(path.join(root,name),batch,{recursive:true,filter:f=>!['logs','submissions_vulcan.tsv','submission_vulcan.lock','cluster_vulcan.txt'].includes(path.basename(f))});
    fs.copyFileSync(path.join(root,'submit_missing_cells_vulcan.sh'),path.join(tmp,'submit.sh'));
    fs.mkdirSync(path.join(project,'cache/surrogates'),{recursive:true});
    for(const pool of new Set(p.cells.map(c=>c.pool)))fs.symlinkSync(path.join(root,'cache/surrogates',pool),path.join(project,'cache/surrogates',pool));
    fs.mkdirSync(path.join(tmp,'bin'));fs.writeFileSync(path.join(tmp,'activate'),'');fs.writeFileSync(path.join(tmp,'bash_env'),'module() { :; }\n');fs.writeFileSync(path.join(tmp,'queue'),'');
    function command(n,body){const f=path.join(tmp,'bin',n);fs.writeFileSync(f,'#!/bin/bash\nset -eu\n'+body+'\n');fs.chmodSync(f,0o700);}
    command('module',':');command('sinfo','echo gpu:l40s:4');command('scontrol','if [ "$1" = update ]; then printf "%s\\n" "$*" >> "$MOCK_ROOT/updates"; else echo "ClusterName = vulcan_mock"; fi');command('squeue','cat "$MOCK_ROOT/queue"');command('sbatch','node "$MOCK_ROOT/sbatch.js" "$@"');
    fs.writeFileSync(path.join(tmp,'sbatch.js'),`const fs=require('fs'),f=process.env.MOCK_ROOT+'/calls',rows=fs.existsSync(f)?fs.readFileSync(f,'utf8').trim().split('\\n'):[],id=17001+rows.length;fs.appendFileSync(f,JSON.stringify({id,args:process.argv.slice(2)})+'\\n');console.log(id);`);
    const env={...process.env,MOCK_ROOT:tmp,BASH_ENV:path.join(tmp,'bash_env'),SOURCE_ROOT:project,DATA_ROOT:path.join(root,'data'),ENV_ACTIVATE:path.join(tmp,'activate'),PATH:path.join(tmp,'bin')+':'+process.env.PATH};
    function run(){const r=spawnSync('sh',[path.join(tmp,'submit.sh')],{env,encoding:'utf8'});assert.equal(r.status,0,r.stdout+r.stderr);}
    const calls=()=>fs.readFileSync(path.join(tmp,'calls'),'utf8').trim().split('\n').map(JSON.parse);
    run();const first=calls(),ids=new Map();assert.equal(first.length,52);
    for(let i=0;i<52;i++){const c=first[i],t=p.tasks[i],tiny=t.dataset==='TinyImageNet';assert.equal(t.account,tiny?'aip-bghaddar':'aip-yiweilu');const time=t.stage==='all'?'02:00:00':t.stage==='craft'?'05:00:00':'03:00:00';for(const flag of ['--account='+(overrides.get(t.id)||t.account),'--time='+time,'--mem='+t.memory,'--cpus-per-task=1','--gres=gpu:l40s:1'])assert(c.args.includes(flag));assert.equal(c.args.find(a=>a.startsWith('--dependency=')),t.requires.length?'--dependency=afterok:'+t.requires.map(r=>ids.get(r)).join(':'):undefined);ids.set(t.id,c.id);}
    assert.equal(first.filter(c=>c.args.includes('--account=aip-bghaddar')).length,p.tasks.filter(t=>(overrides.get(t.id)||t.account)==='aip-bghaddar').length);
    fs.writeFileSync(path.join(tmp,'queue'),first.map((c,i)=>c.id+' PENDING mc_'+p.tasks[i].id+'\n').join(''));run();assert.equal(calls().length,52);
    const parentIndex=p.tasks.findIndex(t=>t.stage==='craft'),childIndex=p.tasks.findIndex(t=>t.requires.includes(p.tasks[parentIndex].id));
    fs.writeFileSync(path.join(tmp,'queue'),first.filter((c,i)=>i!==parentIndex).map(c=>{const i=first.indexOf(c);return c.id+' PENDING mc_'+p.tasks[i].id+'\n';}).join(''));run();assert.equal(calls().length,53);assert(fs.readFileSync(path.join(tmp,'updates'),'utf8').includes('JobId='+first[childIndex].id+' Dependency=afterok:'+calls()[52].id));
    console.log('PASS: ten source-matched 4x4 cells; original account assignments and explicit overrides; correct limits, dependencies, duplicate protection and retry dependency repair.');
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
