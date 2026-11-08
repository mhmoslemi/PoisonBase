// Check source settings and mock only the two corrected submissions.
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert'),crypto=require('crypto');
const {spawnSync}=require('child_process');
const root=path.resolve(__dirname,'../..'),name='sbatch/rebuttal_basis_resnet_fix_vulcan';
const p=JSON.parse(fs.readFileSync(path.join(root,name,'plan.json'))),old=JSON.parse(fs.readFileSync(path.join(root,'sbatch/rebuttal_basis_lambda_killarney/plan.json')));
assert.equal(p.cells.length,2);assert.equal(p.tasks.length,2);
assert.deepEqual(p.tasks.map(t=>t.replaces_job).sort(),['2182295','2182559']);
for(const c of p.cells){assert.deepEqual(c.args,old.cells.find(x=>x.id===c.id).args);assert.equal(c.args.num_surrogates,5);assert.equal(c.args.craft_ensemble,5);assert.equal(c.targets.length,8);assert.equal(c.args.num_victims,5);assert(!Object.hasOwn(c.args,'sel_mode'));}
const w=fs.readFileSync(path.join(root,name,'code/experiments/rebuttal_mr/run.py'),'utf8');
assert(w.includes("if args.get('sel_mode'):"));assert(w.includes("count = cell['args'].get('sel_K') or cell['args']['num_surrogates']"));
assert.equal(fs.readFileSync(path.join(root,name,'surrogates.tsv'),'utf8').trim().split('\n').length,6);
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'basis-resfix-check-')),batch=path.join(tmp,name);
fs.mkdirSync(path.dirname(batch),{recursive:true});fs.cpSync(path.join(root,name),batch,{recursive:true,filter:f=>!['logs','submissions_vulcan.tsv','submission_vulcan.lock','cluster_vulcan.txt'].includes(path.basename(f))});
fs.copyFileSync(path.join(root,'submit_basis_resnet_fix_vulcan_ghaddar.sh'),path.join(tmp,'submit.sh'));
fs.mkdirSync(path.join(tmp,'project/cache/surrogates'),{recursive:true});
fs.symlinkSync(path.join(root,'cache/surrogates',p.cells[0].pool),path.join(tmp,'project/cache/surrogates',p.cells[0].pool));
fs.mkdirSync(path.join(tmp,'bin'));fs.writeFileSync(path.join(tmp,'activate'),'');fs.writeFileSync(path.join(tmp,'bash_env'),'module() { :; }\n');fs.writeFileSync(path.join(tmp,'queue'),'2182559 PENDING bl_c10_res_gm_b002_lam0p1\n');
function command(n,body){const f=path.join(tmp,'bin',n);fs.writeFileSync(f,'#!/bin/bash\nset -eu\n'+body+'\n');fs.chmodSync(f,0o700);}
command('module',':');command('sinfo','printf "gpu:l40s:4\\n"');command('scontrol','printf "ClusterName = vulcan_mock\\n"');command('squeue','cat "$MOCK_ROOT/queue"');command('sbatch','node "$MOCK_ROOT/sbatch.js" "$@"');
fs.writeFileSync(path.join(tmp,'sbatch.js'),`const fs=require('fs'),f=process.env.MOCK_ROOT+'/calls';const rows=fs.existsSync(f)?fs.readFileSync(f,'utf8').trim().split('\\n'):[];const id=13001+rows.length;fs.appendFileSync(f,JSON.stringify({id,args:process.argv.slice(2)})+'\\n');console.log(id);`);
const env={...process.env,MOCK_ROOT:tmp,BASH_ENV:path.join(tmp,'bash_env'),SOURCE_ROOT:path.join(tmp,'project'),DATA_ROOT:path.join(root,'data'),ENV_ACTIVATE:path.join(tmp,'activate'),PATH:path.join(tmp,'bin')+':'+process.env.PATH};
function run(){return spawnSync('sh',[path.join(tmp,'submit.sh')],{env,encoding:'utf8'});}
function calls(){return fs.readFileSync(path.join(tmp,'calls'),'utf8').trim().split('\n').map(JSON.parse);}
try{
    const blocked=run();assert.notEqual(blocked.status,0);assert(blocked.stderr.includes('already active outside this ledger'));assert(!fs.existsSync(path.join(tmp,'calls')));
    fs.writeFileSync(path.join(tmp,'queue'),'');let r=run();assert.equal(r.status,0,r.stdout+r.stderr);assert.equal(calls().length,2);
    for(const c of calls())for(const flag of ['--account=aip-bghaddar','--time=02:00:00','--mem=6G','--cpus-per-task=1','--gres=gpu:l40s:1'])assert(c.args.includes(flag));
    fs.writeFileSync(path.join(tmp,'queue'),calls().map((c,i)=>c.id+' PENDING bl_'+p.tasks[i].id+'\n').join(''));r=run();assert.equal(r.status,0,r.stdout+r.stderr);assert.equal(calls().length,2);
    const done=path.join(env.SOURCE_ROOT,p.results,'jobs',p.tasks[0].id);fs.mkdirSync(done,{recursive:true});const hash=crypto.createHash('sha256').update(fs.readFileSync(path.join(batch,'plan.json'))).update(fs.readFileSync(path.join(batch,'code.sha256'))).digest('hex');fs.writeFileSync(path.join(done,'.done'),JSON.stringify({protocol_hash:hash}));
    fs.writeFileSync(path.join(tmp,'queue'),'13002 PENDING bl_'+p.tasks[1].id+'\n');r=run();assert.equal(r.status,0,r.stdout+r.stderr);assert(r.stdout.includes('complete'));assert.equal(calls().length,2);
    console.log('PASS: two corrected Ghaddar jobs, original 8x5 settings, five configured surrogates, optional sel_mode, duplicate guard and retry/skip.');
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
