// Mock the ten pending-task replacements; neither running CIFAR job is selected.
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert');
const {spawnSync}=require('child_process');
const root=path.resolve(__dirname,'../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'basis-pending-lu-'));
const rel='sbatch/rebuttal_basis_lambda_extra_vulcan',batch=path.join(tmp,rel);
fs.mkdirSync(path.dirname(batch),{recursive:true});fs.cpSync(path.join(root,rel),batch,{recursive:true,filter:f=>!['logs','submissions_vulcan.tsv','submission_vulcan.lock','cluster_vulcan.txt'].includes(path.basename(f))});
fs.copyFileSync(path.join(root,'submit_basis_pending_lu_vulcan.sh'),path.join(tmp,'submit.sh'));
const p=JSON.parse(fs.readFileSync(path.join(batch,'plan.json'))),tasks=p.tasks.filter(t=>t.account==='aip-boyuwang'&&t.dataset!=='CIFAR10');assert.equal(tasks.length,10);
const project=path.join(tmp,'project'),pools=path.join(project,'cache/surrogates');fs.mkdirSync(pools,{recursive:true});
for(const pool of new Set(p.cells.map(c=>c.pool)))fs.symlinkSync(path.join(root,'cache/surrogates',pool),path.join(pools,pool));
fs.mkdirSync(path.join(tmp,'bin'));fs.writeFileSync(path.join(tmp,'activate'),'');fs.writeFileSync(path.join(tmp,'queue'),'');fs.writeFileSync(path.join(tmp,'bash_env'),'module() { :; }\n');
function command(n,body){const f=path.join(tmp,'bin',n);fs.writeFileSync(f,'#!/bin/bash\nset -eu\n'+body+'\n');fs.chmodSync(f,0o700);}
command('module',':');command('sinfo','printf "gpu:l40s:4\\n"');command('scontrol','printf "ClusterName = vulcan_mock\\n"');command('squeue','cat "$MOCK_ROOT/queue"');command('sbatch','node "$MOCK_ROOT/sbatch.js" "$@"');
fs.writeFileSync(path.join(tmp,'sbatch.js'),`const fs=require('fs'),f=process.env.MOCK_ROOT+'/calls',rows=fs.existsSync(f)?fs.readFileSync(f,'utf8').trim().split('\\n'):[],id=14001+rows.length;fs.appendFileSync(f,JSON.stringify({id,args:process.argv.slice(2)})+'\\n');console.log(id);`);
const env={...process.env,MOCK_ROOT:tmp,BASH_ENV:path.join(tmp,'bash_env'),SOURCE_ROOT:project,ENV_ACTIVATE:path.join(tmp,'activate'),DATA_ROOT:path.join(root,'data'),PATH:path.join(tmp,'bin')+':'+process.env.PATH};
function run(){const r=spawnSync('sh',[path.join(tmp,'submit.sh')],{env,encoding:'utf8'});assert.equal(r.status,0,r.stdout+r.stderr);}
function calls(){return fs.readFileSync(path.join(tmp,'calls'),'utf8').trim().split('\n').map(JSON.parse);}
try{
    run();const first=calls(),ids=new Map();assert.equal(first.length,10);
    for(let i=0;i<10;i++){const c=first[i],t=tasks[i];for(const flag of ['--account=aip-yiweilu','--gres=gpu:l40s:1','--time=06:00:00','--cpus-per-task=1','--mem='+t.memory])assert(c.args.includes(flag));assert(c.args.includes(t.id));const dep=c.args.find(a=>a.startsWith('--dependency='));assert.equal(dep,t.requires.length?'--dependency=afterok:'+t.requires.map(p=>ids.get(p)).join(':'):undefined);ids.set(t.id,c.id);}
    assert(first.every(c=>!c.args.some(a=>a.includes('bl_c10_'))));
    fs.writeFileSync(path.join(tmp,'queue'),first.map((c,i)=>c.id+' PENDING bl_'+tasks[i].id+'\n').join(''));run();assert.equal(calls().length,10);
    console.log('PASS: exactly ten Lu jobs, original limits/dependencies, no running CIFAR tasks, duplicate-safe rerun.');
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
