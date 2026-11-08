// No Slurm calls or Python: exercise submission, dependency repair, and resume.
const fs = require('fs');
const path = require('path');
const os = require('os');
const assert = require('assert');
const {spawnSync} = require('child_process');
const root = path.resolve(__dirname, '../..');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'mr-submit-check-'));
const batch = path.join(tmp, 'sbatch/rebuttal_mr_killarney');
fs.mkdirSync(path.dirname(batch), {recursive: true});
fs.cpSync(path.join(root, 'sbatch/rebuttal_mr_killarney'), batch, {recursive: true,
    filter: p => !['logs', 'submission.lock', 'submissions.tsv', 'cluster.txt'].includes(path.basename(p))});
fs.copyFileSync(path.join(root, 'submit_mr_killarney_ghaddar.sh'), path.join(tmp, 'submit.sh'));
fs.mkdirSync(path.join(tmp, 'bin'));
fs.mkdirSync(path.join(tmp, 'project'));
fs.symlinkSync(path.join(root, 'cache'), path.join(tmp, 'project/cache'));
fs.writeFileSync(path.join(tmp, 'activate'), '');
fs.writeFileSync(path.join(tmp, 'queue'), '');
fs.writeFileSync(path.join(tmp, 'counter'), '8000');
function command(name, body) {
    const f = path.join(tmp, 'bin', name);
    fs.writeFileSync(f, '#!/bin/bash\nset -eu\n' + body + '\n');
    fs.chmodSync(f, 0o700);
}
command('sbatch', 'node "$MOCK_ROOT/sbatch.js" "$@"');
command('squeue', 'cat "$MOCK_ROOT/queue"');
command('sinfo', 'printf "gpu:l40s:4\\n"');
command('scontrol', 'if [ "$1" = show ]; then printf "ClusterName = killarney_mock\\n"; else printf "%s " "$@" >> "$MOCK_ROOT/updates"; printf "\\n" >> "$MOCK_ROOT/updates"; fi');
fs.writeFileSync(path.join(tmp, 'sbatch.js'), `
const fs=require('fs'),p=process.env.MOCK_ROOT;
const id=Number(fs.readFileSync(p+'/counter','utf8'))+1;
fs.writeFileSync(p+'/counter',String(id));
fs.appendFileSync(p+'/calls',JSON.stringify({id,args:process.argv.slice(2)})+'\\n');
console.log(id);
`);
const env = {...process.env, MOCK_ROOT: tmp, SOURCE_ROOT: path.join(tmp, 'project'),
    DATA_ROOT: path.join(root, 'data'), ENV_ACTIVATE: path.join(tmp, 'activate'),
    PATH: path.join(tmp, 'bin') + ':' + process.env.PATH};
function run() {
    const result = spawnSync('sh', [path.join(tmp, 'submit.sh')], {env, encoding: 'utf8'});
    assert.equal(result.status, 0, result.stdout + result.stderr);
    return result.stdout;
}
function calls() {
    return fs.readFileSync(path.join(tmp, 'calls'), 'utf8').trim().split('\n').map(JSON.parse);
}
function queue(ids) {
    fs.writeFileSync(path.join(tmp, 'queue'), ids.map(id => id + ' PENDING\n').join(''));
}
try {
    run();
    const initial = calls();
    assert.equal(initial.length, 29);
    const plan = JSON.parse(fs.readFileSync(path.join(batch, 'plan.json')));
    const ids = new Map();
    for (let i = 0; i < initial.length; i++) {
        const {id, args} = initial[i], task = plan.tasks[i];
        assert(args.includes('--account=aip-bghaddar'));
        assert(args.includes('--gres=gpu:1'));
        assert(args.includes('--cpus-per-task=1'));
        assert(args.includes('--mem=' + task.memory));
        assert(args.includes('--job-name=mrk_' + task.id));
        if (task.dataset === 'TinyImageNet') assert(args.includes('--time=04:00:00'));
        const dep = args.find(a => a.startsWith('--dependency='));
        if (task.requires.length) assert.equal(dep, '--dependency=afterok:' + task.requires.map(p => ids.get(p)).join(':'));
        else assert.equal(dep, undefined);
        ids.set(task.id, id);
    }
    queue(initial.map(c => c.id));
    run();
    assert.equal(calls().length, 29, 'Rerunning must not duplicate active jobs');
    queue(initial.slice(1).map(c => c.id));
    run();
    assert.equal(calls().length, 30, 'Only failed preflight should be resubmitted');
    const repaired = fs.readFileSync(path.join(tmp, 'updates'), 'utf8').trim().split('\n');
    assert.equal(repaired.length, 28);
    assert(repaired.every(s => s.includes('afterok:8030')));
    const craftTask = plan.tasks.find(t => t.stage === 'craft');
    const oldCraft = ids.get(craftTask.id);
    queue([...initial.slice(1).map(c => c.id).filter(id => id !== oldCraft), 8030]);
    run();
    assert.equal(calls().length, 31, 'Only failed crafting should be resubmitted');
    const updates = fs.readFileSync(path.join(tmp, 'updates'), 'utf8').trim().split('\n');
    assert.equal(updates.length, 30, 'Both downstream evaluations must depend on replacement craft');
    assert(updates.slice(-2).every(s => s.includes('afterok:8030:8031')));
    // A completed result should be skipped even after its job leaves squeue.
    const crypto = require('crypto');
    const hash = crypto.createHash('sha256').update(fs.readFileSync(path.join(batch, 'plan.json')))
        .update(fs.readFileSync(path.join(batch, 'code.sha256'))).digest('hex');
    const done = path.join(tmp, 'project/rebuttal_mr_result/jobs', plan.tasks[1].id);
    fs.mkdirSync(done, {recursive: true});
    fs.writeFileSync(path.join(done, '.done'), JSON.stringify({protocol_hash: hash}));
    queue([...initial.slice(2).map(c => c.id).filter(id => id !== oldCraft), 8030, 8031]);
    assert(run().includes('SKIP ' + plan.tasks[1].id + ': complete'));
    assert.equal(calls().length, 31);
    console.log('PASS: 29 jobs, exact resources/dependencies, no duplicate submissions, failed-dependency repair, completed-task skip.');
} finally {
    fs.rmSync(tmp, {recursive: true, force: true});
}
