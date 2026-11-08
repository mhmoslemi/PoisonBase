// Mock Slurm: validate resume selection/resources without running training.
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert');
const {spawnSync}=require('child_process'),root=path.resolve(__dirname,'../..');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'w5-resume-'));
try{
    const batch='sbatch/rebuttal_w5_resume_vulcan';fs.mkdirSync(path.join(tmp,batch),{recursive:true});
    fs.copyFileSync(path.join(root,batch,'tasks.tsv'),path.join(tmp,batch,'tasks.tsv'));
    fs.copyFileSync(path.join(root,'submit_w5_resume_vulcan_ghaddar.sh'),path.join(tmp,'submit.sh'));
    fs.symlinkSync(path.join(root,'sbatch/rebuttal_w5_killarney'),path.join(tmp,'sbatch/rebuttal_w5_killarney'));
    for(const dir of ['data','rebuttal_w5_result'])fs.symlinkSync(path.join(root,dir),path.join(tmp,dir));
    fs.mkdirSync(path.join(tmp,'bin'));fs.writeFileSync(path.join(tmp,'queue'),'');
    const command=(name,body)=>{const f=path.join(tmp,'bin',name);fs.writeFileSync(f,'#!/bin/bash\nset -eu\n'+body+'\n');fs.chmodSync(f,0o700);};
    command('scontrol','echo "ClusterName = vulcan_mock"');command('sinfo','echo gpu:l40s:4');command('squeue','cat "$MOCK_ROOT/queue"');
    command('sbatch','node "$MOCK_ROOT/sbatch.js" "$@"');
    fs.writeFileSync(path.join(tmp,'sbatch.js'),`const fs=require('fs'),f=process.env.MOCK_ROOT+'/calls',rows=fs.existsSync(f)?fs.readFileSync(f,'utf8').trim().split('\\n'):[],id=16001+rows.length;fs.appendFileSync(f,JSON.stringify({id,args:process.argv.slice(2)})+'\\n');console.log(id);`);
    const env={...process.env,MOCK_ROOT:tmp,PATH:path.join(tmp,'bin')+':'+process.env.PATH};
    const run=()=>{const r=spawnSync('sh',[path.join(tmp,'submit.sh')],{env,encoding:'utf8'});assert.equal(r.status,0,r.stdout+r.stderr);};
    run();const calls=fs.readFileSync(path.join(tmp,'calls'),'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(calls.length,22);let four=0;
    for(const c of calls){for(const flag of ['--account=aip-bghaddar','--cpus-per-task=1','--mem=6G','--gres=gpu:l40s:1'])assert(c.args.includes(flag));const task=c.args.at(-2),seconds=+c.args.at(-1);assert.equal(seconds,['at_res_12','at_res_21'].includes(task)?14400:10800);assert(c.args.includes('--time=0'+seconds/3600+':00:00'));if(seconds===14400)four++;assert(!task.startsWith('ac_'));}
    assert.equal(four,2);fs.writeFileSync(path.join(tmp,'queue'),calls.map(c=>c.id+' '+c.args.find(a=>a.startsWith('--job-name=')).slice(11)+'\n').join(''));
    run();assert.equal(fs.readFileSync(path.join(tmp,'calls'),'utf8').trim().split('\n').length,22);
    console.log('PASS: 22 unfinished tasks only, Ghaddar/6G/one L40S, 20 three-hour and two four-hour jobs, no duplicate submissions.');
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
