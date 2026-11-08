// Validate the exact moved half and mock Fir submission/staging without GPU work.
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert'),crypto=require('crypto');
const {spawnSync}=require('child_process');
const root=path.resolve(__dirname,'../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'basis-fir-check-'));
const name='sbatch/rebuttal_basis_lambda_fir',batch=path.join(tmp,name);
fs.mkdirSync(path.dirname(batch),{recursive:true});
fs.cpSync(path.join(root,name),batch,{recursive:true,filter:p=>!['logs','submissions_fir.tsv','submission_fir.lock','cluster_fir.txt'].includes(path.basename(p))});
fs.copyFileSync(path.join(root,'submit_basis_lambda_fir_ghaddar.sh'),path.join(tmp,'submit_basis_lambda_fir_ghaddar.sh'));
const plan=JSON.parse(fs.readFileSync(path.join(batch,'plan.json')));
const original=JSON.parse(fs.readFileSync(path.join(root,'sbatch/rebuttal_basis_lambda_killarney/plan.json')));
const relabel=items=>items.map(x=>({...x,site:'fir'}));
assert.deepEqual(plan.cells,relabel(original.cells.filter(c=>c.site==='killarney')));
assert.deepEqual(plan.tasks,relabel(original.tasks.filter(t=>t.site==='killarney')));
assert.equal(plan.tasks.length,33);assert.equal(plan.cells.length,12);
assert.equal(plan.cells.reduce((n,c)=>n+c.targets.length*c.args.num_victims,0),264);
assert.equal(plan.results,'rebuttal_basis_lambda_fir_result');
const pools=path.join(tmp,'project/cache/surrogates');fs.mkdirSync(pools,{recursive:true});
for(const pool of new Set(plan.cells.map(c=>c.pool)))fs.symlinkSync(path.join(root,'cache/surrogates',pool),path.join(pools,pool));
fs.mkdirSync(path.join(tmp,'bin'));fs.writeFileSync(path.join(tmp,'queue'),'');fs.writeFileSync(path.join(tmp,'activate'),'');fs.writeFileSync(path.join(tmp,'counter'),'12000');
function command(n,body){const f=path.join(tmp,'bin',n);fs.writeFileSync(f,'#!/bin/bash\nset -eu\n'+body+'\n');fs.chmodSync(f,0o700);}
command('module',':');command('sinfo','printf "gpu:h100:4\\n"');command('squeue','cat "$MOCK_ROOT/queue"');
fs.writeFileSync(path.join(tmp,'bash_env'),'module() { :; }\n');
command('scontrol','if [ "$1" = show ]; then printf "ClusterName = fir_mock\\n"; else printf "%s\\n" "$*" >> "$MOCK_ROOT/updates"; fi');
command('sbatch','node "$MOCK_ROOT/sbatch.js" "$@"');
fs.writeFileSync(path.join(tmp,'sbatch.js'),`const fs=require('fs'),p=process.env.MOCK_ROOT,id=Number(fs.readFileSync(p+'/counter'))+1;fs.writeFileSync(p+'/counter',String(id));fs.appendFileSync(p+'/calls',JSON.stringify({id,args:process.argv.slice(2)})+'\n');console.log(id);`.replace("+'\n'","+'\\n'"));
const env={...process.env,BASH_ENV:path.join(tmp,'bash_env'),MOCK_ROOT:tmp,SOURCE_ROOT:path.join(tmp,'project'),ENV_ACTIVATE:path.join(tmp,'activate'),DATA_ROOT:path.join(root,'data'),PATH:path.join(tmp,'bin')+':'+process.env.PATH};
function run(){const r=spawnSync('sh',[path.join(tmp,'submit_basis_lambda_fir_ghaddar.sh')],{env,encoding:'utf8'});assert.equal(r.status,0,r.stdout+r.stderr);return r.stdout;}
function calls(){return fs.readFileSync(path.join(tmp,'calls'),'utf8').trim().split('\n').map(JSON.parse);}
function queue(ids){fs.writeFileSync(path.join(tmp,'queue'),ids.map(i=>i+' PENDING\n').join(''));}
try{
    run();const initial=calls(),ids=new Map();assert.equal(initial.length,33);
    for(let i=0;i<33;i++){
        const t=plan.tasks[i],c=initial[i];assert(t.seconds<=21600);
        assert(c.args.includes('--account=def-bghaddar'));assert(c.args.includes('--gpus-per-node=h100:1'));
        assert(c.args.includes('--cpus-per-task=1'));assert(c.args.includes('--mem='+t.memory));
        assert(c.args.includes('--time='+[Math.floor(t.seconds/3600),Math.floor(t.seconds/60)%60,t.seconds%60].map(n=>String(n).padStart(2,'0')).join(':')));
        const dep=c.args.find(a=>a.startsWith('--dependency='));
        assert.equal(dep,t.requires.length?'--dependency=afterok:'+t.requires.map(p=>ids.get(p)).join(':'):undefined);
        ids.set(t.id,c.id);
    }
    queue(initial.map(c=>c.id));run();assert.equal(calls().length,33);
    const craft=plan.tasks.find(t=>t.stage==='craft'),old=ids.get(craft.id);
    queue(initial.map(c=>c.id).filter(i=>i!==old));run();assert.equal(calls().length,34);
    assert(fs.readFileSync(path.join(tmp,'updates'),'utf8').includes('12034'));
    const done=path.join(env.SOURCE_ROOT,plan.results,'jobs',plan.tasks[0].id);
    fs.mkdirSync(done,{recursive:true});const hash=crypto.createHash('sha256').update(fs.readFileSync(path.join(batch,'plan.json'))).update(fs.readFileSync(path.join(batch,'code.sha256'))).digest('hex');
    fs.writeFileSync(path.join(done,'.done'),JSON.stringify({protocol_hash:hash}));
    queue([...initial.map(c=>c.id).filter(i=>i!==old&&i!==initial[0].id),12034]);
    assert(run().includes('SKIP '+plan.tasks[0].id+': complete'));assert.equal(calls().length,34);
    const realSha=spawnSync('which',['sha256sum'],{encoding:'utf8'}).stdout.trim();
    command('sha256sum','if [ "$PWD" = "$MOCK_BATCH" ]; then exec "$REAL_SHA" "$@"; else cat >/dev/null; fi');
    command('rsync','[ -s "$2" ]; printf "%s\\n" "$2" >> "$MOCK_ROOT/staged"');command('srun',':');
    const t=plan.tasks.find(t=>t.dataset==='TinyImageNet');
    const r=spawnSync('bash',[path.join(batch,'job.sh'),env.SOURCE_ROOT,batch,t.id,env.ENV_ACTIVATE,env.DATA_ROOT,'fir'],{env:{...env,MOCK_BATCH:batch,REAL_SHA:realSha,SLURM_TMPDIR:path.join(tmp,'node')},encoding:'utf8'});
    assert.equal(r.status,0,r.stdout+r.stderr);
    const staged=fs.readFileSync(path.join(tmp,'staged'),'utf8').trim().split('\n').filter(f=>f.includes('/cache/surrogates/'));
    assert.equal(staged.length,20);assert(staged.every(f=>!f.includes('_new/')));
    console.log('PASS: exact 33-job Fir replacement, def-bghaddar/H100, unchanged <=6h resources, original pool paths, dependencies and retry/skip.');
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
