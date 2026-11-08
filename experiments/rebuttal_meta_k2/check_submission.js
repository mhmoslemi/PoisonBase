// No Python/GPU work: check the six-cell plan and exercise mocked Slurm.
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert');
const {spawnSync}=require('child_process');
const root=path.resolve(__dirname,'../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'meta-k2-check-'));
const batch=path.join(tmp,'sbatch/rebuttal_meta_k2_vulcan');
fs.mkdirSync(path.dirname(batch),{recursive:true});
fs.cpSync(path.join(root,'sbatch/rebuttal_meta_k2_vulcan'),batch,{recursive:true,
    filter:p=>!['logs','submissions.tsv','submission.lock','cluster.txt'].includes(path.basename(p))});
fs.copyFileSync(path.join(root,'submit_meta_k2_vulcan_lu.sh'),path.join(tmp,'submit.sh'));
const plan=JSON.parse(fs.readFileSync(path.join(batch,'plan.json')));
assert.equal(plan.tasks.length,6);
assert.equal(plan.protocol.surrogates,2);
assert.deepEqual(plan.protocol.selection_surrogate_ids,[0,1]);
assert.equal(plan.protocol.crafting.ensemble,5);
assert.equal(plan.protocol.victim.runs,5);
const original=JSON.parse(fs.readFileSync(path.join(root,'sbatch/rebuttal_meta_influence/plan.json')));
for(const t of plan.tasks){
    assert.equal(t.time,'03:00:00');assert.equal(t.stage,'evaluate');assert.equal(t.method,'meta');
    assert.equal(t.targets.length,8);assert.deepEqual(t.dependencies,[]);
    const source=original.tasks.find(s=>s.id===t.source_job);
    assert.deepEqual(t.targets,source.targets);
    assert.equal(t.model,source.model);assert.equal(t.attack,source.attack);assert.equal(t.budget,source.budget);
}
const worker=fs.readFileSync(path.join(batch,'code/experiments/rebuttal_selection/worker.py'),'utf8');
assert(worker.includes("'--sel_K', '2'"));
assert(worker.includes("'--num_surrogates', '5'"));
assert(worker.includes("'--craft_ensemble', '5'"));
assert(worker.includes("'--num_victims', '5'"));
assert(worker.includes("protocol['score_source_hash']"));
assert(worker.includes("source = results.parent / 'rebuttal_meta_influence_result'"));
const inputs=fs.readFileSync(path.join(batch,'inputs.tsv'),'utf8').trim().split('\n').slice(1).map(l=>l.split('\t')[0]);
assert.equal(inputs.filter(f=>f.includes('/scores/')).length,64);
const project=path.join(tmp,'project');
for(const relative of inputs){
    const dest=path.join(project,relative);fs.mkdirSync(path.dirname(dest),{recursive:true});
    fs.symlinkSync(path.join(root,relative),dest);
}
fs.mkdirSync(path.join(tmp,'bin'));fs.writeFileSync(path.join(tmp,'queue'),'');fs.writeFileSync(path.join(tmp,'activate'),'');
function command(name,body){const f=path.join(tmp,'bin',name);fs.writeFileSync(f,'#!/bin/bash\nset -eu\n'+body+'\n');fs.chmodSync(f,0o700);}
command('sinfo','printf "gpu:l40s:4\\n"');
command('scontrol','printf "ClusterName = vulcan_mock\\n"');
command('squeue','cat "$MOCK_ROOT/queue"');
command('sbatch','node "$MOCK_ROOT/sbatch.js" "$@"');
fs.writeFileSync(path.join(tmp,'sbatch.js'),`
const fs=require('fs'),p=process.env.MOCK_ROOT;
const f=p+'/calls',rows=fs.existsSync(f)?fs.readFileSync(f,'utf8').trim().split('\\n'):[];
const id=10001+rows.length;fs.appendFileSync(f,JSON.stringify({id,args:process.argv.slice(2)})+'\\n');console.log(id);
`);
const env={...process.env,MOCK_ROOT:tmp,SOURCE_ROOT:project,ENV_ACTIVATE:path.join(tmp,'activate'),PATH:path.join(tmp,'bin')+':'+process.env.PATH};
function run(){const r=spawnSync('sh',[path.join(tmp,'submit.sh')],{env,encoding:'utf8'});assert.equal(r.status,0,r.stdout+r.stderr);return r.stdout;}
function calls(){return fs.readFileSync(path.join(tmp,'calls'),'utf8').trim().split('\n').map(JSON.parse);}
function queue(ids){fs.writeFileSync(path.join(tmp,'queue'),ids.map(id=>id+' RUNNING\n').join(''));}
try{
    const score=path.join(project,inputs.find(f=>f.includes('/scores/')));
    fs.renameSync(score,score+'.absent');
    const missing=spawnSync('sh',[path.join(tmp,'submit.sh')],{env,encoding:'utf8'});
    assert.notEqual(missing.status,0);assert(missing.stderr.includes('missing input:'),missing.stdout+missing.stderr);assert(!fs.existsSync(path.join(tmp,'calls')));
    fs.renameSync(score+'.absent',score);
    run();const initial=calls();assert.equal(initial.length,6);
    for(let i=0;i<6;i++){
        const a=initial[i].args;for(const flag of ['--account=aip-yiweilu','--time=03:00:00','--cpus-per-task=1','--mem=6G','--gres=gpu:l40s:1'])assert(a.includes(flag));
        assert(a.includes(plan.tasks[i].id));assert(!a.some(s=>s.startsWith('--dependency')));
    }
    queue(initial.map(c=>c.id));run();assert.equal(calls().length,6);
    queue(initial.slice(1).map(c=>c.id));run();assert.equal(calls().length,7);
    const done=path.join(project,'rebuttal_meta_k2_result/jobs',plan.tasks[0].id);fs.mkdirSync(done,{recursive:true});
    fs.writeFileSync(path.join(done,'.done'),'complete\n');fs.writeFileSync(path.join(done,'task.json'),JSON.stringify({task:plan.tasks[0]}));
    assert(run().includes('SKIP '+plan.tasks[0].id+': complete'));assert.equal(calls().length,7);
    console.log('PASS: six Lu jobs, 3h/6G/1CPU/L40S, exact 8x5 grids, two cached selection surrogates, five crafting surrogates, input checks and retry/skip.');
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
