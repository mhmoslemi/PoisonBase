// Exercise submission without Slurm, data loading, or training.
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert');
const {spawnSync}=require('child_process');
const root=path.resolve(__dirname,'../..'),tmp=fs.mkdtempSync(path.join(os.tmpdir(),'mr-victims-'));
const original='sbatch/rebuttal_mr_killarney',wrapper='sbatch/rebuttal_mr_vulcan_victims';
fs.mkdirSync(path.join(tmp,wrapper),{recursive:true});
fs.symlinkSync(path.join(root,original),path.join(tmp,original));
fs.copyFileSync(path.join(root,'submit_mr_tiny_gm_victims_vulcan_ghaddar.sh'),path.join(tmp,'submit.sh'));
const plan=JSON.parse(fs.readFileSync(path.join(root,original,'plan.json')));
const tasks=plan.tasks.filter(t=>t.stage==='victims'&&['tiny_gm_M','tiny_gm_R'].includes(t.cell));
assert.equal(tasks.length,8);
fs.mkdirSync(path.join(tmp,'data'));fs.writeFileSync(path.join(tmp,'data/tinyimagenet.pt'),'fixture');
fs.mkdirSync(path.join(tmp,'rebuttal_mr_result'));
fs.copyFileSync(path.join(root,'rebuttal_mr_result/preflight.json'),path.join(tmp,'rebuttal_mr_result/preflight.json'));
const poisons=tasks.map(t=>path.join(tmp,'rebuttal_mr_result/cells',t.cell,'target_'+t.targets[0],'poison.pt'));
for(const file of poisons){fs.mkdirSync(path.dirname(file),{recursive:true});fs.writeFileSync(file,'fixture');}
fs.mkdirSync(path.join(tmp,'bin'));fs.writeFileSync(path.join(tmp,'activate'),'');fs.writeFileSync(path.join(tmp,'queue'),'');
function command(name,body){const file=path.join(tmp,'bin',name);fs.writeFileSync(file,'#!/bin/bash\nset -eu\n'+body+'\n');fs.chmodSync(file,0o700);}
command('scontrol','printf "ClusterName = vulcan_mock\\n"');command('sinfo','printf "gpu:l40s:4\\n"');
command('squeue','cat "$MOCK_ROOT/queue"');command('sbatch','node "$MOCK_ROOT/sbatch.js" "$@"');
fs.writeFileSync(path.join(tmp,'sbatch.js'),`const fs=require('fs'),f=process.env.MOCK_ROOT+'/calls',rows=fs.existsSync(f)?fs.readFileSync(f,'utf8').trim().split('\\n'):[],id=15001+rows.length;fs.appendFileSync(f,JSON.stringify({id,args:process.argv.slice(2)})+'\\n');console.log(id);`);
const env={...process.env,MOCK_ROOT:tmp,ENV_ACTIVATE:path.join(tmp,'activate'),PATH:path.join(tmp,'bin')+':'+process.env.PATH};
function run(){return spawnSync('sh',[path.join(tmp,'submit.sh')],{env,encoding:'utf8'});}
function calls(){return fs.readFileSync(path.join(tmp,'calls'),'utf8').trim().split('\n').map(JSON.parse);}
try{
    let result=run();assert.equal(result.status,0,result.stdout+result.stderr);
    const first=calls();assert.equal(first.length,8);
    for(let i=0;i<8;i++){
        const args=first[i].args;
        for(const flag of ['--account=aip-bghaddar','--time=04:00:00','--mem=18G','--cpus-per-task=1','--gres=gpu:l40s:1',tasks[i].id])assert(args.includes(flag));
        assert(!args.some(a=>a.startsWith('--dependency=')));
    }
    fs.writeFileSync(path.join(tmp,'queue'),first.map((c,i)=>c.id+' PENDING mrv_'+tasks[i].id+'\n').join(''));
    result=run();assert.equal(result.status,0,result.stdout+result.stderr);assert.equal(calls().length,8);
    fs.unlinkSync(poisons[7]);result=run();assert.notEqual(result.status,0);assert(result.stderr.includes('missing completed poison'));assert.equal(calls().length,8);
    console.log('PASS: eight victim-only Ghaddar jobs, correct resources, active-job reuse, missing-poison guard.');
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
