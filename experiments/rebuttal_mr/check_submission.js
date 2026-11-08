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
const pools = path.join(tmp, 'project/cache/surrogates');
fs.mkdirSync(pools, {recursive: true});
for (const pool of new Set(JSON.parse(fs.readFileSync(path.join(batch, 'plan.json'))).cells.map(c => c.pool))) {
    fs.symlinkSync(path.join(root, 'cache/surrogates', pool),
        path.join(pools, pool + (pool.startsWith('ConvNet') ? '' : '_new')));
}
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
    const tinyNew = path.join(pools, 'TinyImageNet_ResNet18BN_60ep_lr0.1_bs128_seed42_new');
    fs.renameSync(tinyNew, tinyNew.slice(0, -4));
    const missingNew = spawnSync('sh', [path.join(tmp, 'submit.sh')], {env, encoding: 'utf8'});
    assert.notEqual(missingNew.status, 0);
    assert(missingNew.stderr.includes('seed42_new/net_0.pt'));
    assert(!fs.existsSync(path.join(tmp, 'calls')), 'Missing _new pool must fail before submission');
    fs.renameSync(tinyNew.slice(0, -4), tinyNew);
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
    // Exercise staging with mocked module/rsync/srun; no Python or file copies.
    const realSha = spawnSync('which', ['sha256sum'], {encoding: 'utf8'}).stdout.trim();
    fs.writeFileSync(path.join(tmp, 'bash_env'), 'module() { :; }\n');
    command('sha256sum', 'if [ "$PWD" = "$MOCK_BATCH" ]; then exec "$REAL_SHA" "$@"; else cat >/dev/null; fi');
    command('rsync', '[ -s "$2" ]; printf "%s\\t%s\\n" "$2" "$3" >> "$MOCK_ROOT/staged"');
    command('srun', 'printf "mocked GPU worker\\n"');
    const stageEnv = {...env, BASH_ENV: path.join(tmp, 'bash_env'), MOCK_BATCH: batch,
        REAL_SHA: realSha, SLURM_TMPDIR: path.join(tmp, 'node_tmp')};
    const staged = spawnSync('bash', [path.join(batch, 'job.sh'), env.SOURCE_ROOT, batch,
        'preflight', env.ENV_ACTIVATE, env.DATA_ROOT], {env: stageEnv, encoding: 'utf8'});
    assert.equal(staged.status, 0, staged.stdout + staged.stderr);
    const sources = fs.readFileSync(path.join(tmp, 'staged'), 'utf8').trim().split('\n').map(s => s.split('\t')[0]);
    const surrogateSources = sources.filter(s => s.includes('/cache/surrogates/'));
    assert.equal(surrogateSources.length, 60);
    assert.equal(surrogateSources.filter(s => s.includes('seed42_new/')).length, 40);
    assert(surrogateSources.filter(s => !s.includes('seed42_new/')).every(s => s.includes('/ConvNetBN_')));
    console.log('PASS: 29 jobs, resources/dependencies/resume, required _new pools without fallback, staging 40 ResNets from _new and 20 ConvNets from original.');
} finally {
    fs.rmSync(tmp, {recursive: true, force: true});
}
