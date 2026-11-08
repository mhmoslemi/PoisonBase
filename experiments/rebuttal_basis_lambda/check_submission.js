// Mock Slurm: validate cell coverage, resource limits, resume and dependencies.
const fs = require('fs');
const path = require('path');
const os = require('os');
const assert = require('assert');
const crypto = require('crypto');
const {spawnSync} = require('child_process');
const root = path.resolve(__dirname, '../..');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'basis-lambda-check-'));
const batch = path.join(tmp, 'sbatch/rebuttal_basis_lambda_killarney');
fs.mkdirSync(path.dirname(batch), {recursive: true});
fs.cpSync(path.join(root, 'sbatch/rebuttal_basis_lambda_killarney'), batch,
    {recursive: true, filter: p => !['logs', 'submission.lock', 'submissions.tsv', 'cluster.txt'].includes(path.basename(p))});
for (const site of ['killarney', 'vulcan']) fs.copyFileSync(path.join(root, `submit_basis_lambda_${site}_ghaddar.sh`), path.join(tmp, `submit_basis_lambda_${site}_ghaddar.sh`));
const plan = JSON.parse(fs.readFileSync(path.join(batch, 'plan.json')));
assert.equal(plan.cells.length, 24);
assert.equal(plan.tasks.length, 66);
for (const site of ['vulcan', 'killarney']) {
    const tasks = plan.tasks.filter(t => t.site === site);
    assert.equal(tasks.length, 33);
    const ids = new Set(tasks.map(t => t.id));
    for (const t of tasks) for (const dep of t.requires) assert(ids.has(dep), 'Cross-cluster dependency');
    assert.equal(plan.cells.filter(c => c.site === site).reduce((n,c) => n+c.targets.length*c.args.num_victims,0),264);
}
assert.equal(plan.cells.reduce((n, c) => n + c.targets.length * c.args.num_victims, 0), 528);
for (const c of plan.cells) {
    const tasks = plan.tasks.filter(t => t.cell === c.id);
    const crafts = tasks.filter(t => t.stage !== 'victims').flatMap(t => t.targets);
    const victims = tasks.filter(t => t.stage !== 'craft').flatMap(t => t.targets);
    assert.deepEqual([...crafts].sort(), [...c.targets].sort());
    assert.deepEqual([...victims].sort(), [...c.targets].sort());
    assert(tasks.every(t => t.site === c.site));
    assert([0.1, 1, 10].includes(c.lambda));
    assert.equal(c.lambda, c.args.lambda_margin);
    if (c.lambda === 1) assert(['tiny_gm_b005', 'tiny_sapa_b005'].includes(c.row));
    assert.equal(c.targets.length, c.args.dataset === 'CIFAR10' ? 8 : 4);
    assert.equal(c.args.num_victims, c.args.dataset === 'CIFAR10' ? 5 : 4);
    assert.equal(c.args.craft_steps, 250);
    assert.equal(c.args.victim_epochs, 50);
    const source = fs.readFileSync(path.join(root, c.source_log), 'utf8');
    const argsLine = source.split('\n').find(l => l.includes('args: '));
    const original = JSON.parse(argsLine.slice(argsLine.indexOf('args: ') + 6));
    const expected = {...original, budget: c.args.budget, lambda_margin: c.lambda,
        num_targets: c.targets.length, num_victims: c.args.num_victims,
        num_poisons: c.args.num_poisons};
    assert.deepEqual(c.args, expected, 'Source settings changed');
    const targetLine = source.split('\n').find(l => l.includes('targets: ['));
    assert.deepEqual(c.targets, JSON.parse(targetLine.slice(targetLine.indexOf('targets: ') + 9)).slice(0, c.targets.length));
}
for (const row of plan.rows) {
    assert.deepEqual(plan.cells.filter(c => c.row === row.id).map(c => c.lambda),
        ['tiny_gm_b005', 'tiny_sapa_b005'].includes(row.id) ? [0.1, 1, 10] : [0.1, 10]);
}
const worker = fs.readFileSync(path.join(batch, 'code/experiments/rebuttal_mr/run.py'), 'utf8');
assert(worker.includes('distance_margin_coef=None'));
assert(worker.includes("sharp_mode=args['sharp_mode'] if args['attack'] == 'sapa' else None"));
assert(worker.includes("resume_state=state['state'] if state else None"));
const pools = path.join(tmp, 'project/cache/surrogates');
fs.mkdirSync(pools, {recursive: true});
for (const pool of new Set(plan.cells.map(c => c.pool))) {
    for (const suffix of /^(CIFAR100|TinyImageNet)_/.test(pool) ? ['', '_new'] : [''])
        fs.symlinkSync(path.join(root, 'cache/surrogates', pool), path.join(pools, pool + suffix));
}
fs.mkdirSync(path.join(tmp, 'bin'));
fs.writeFileSync(path.join(tmp, 'activate'), '');
fs.writeFileSync(path.join(tmp, 'queue'), '');
fs.writeFileSync(path.join(tmp, 'counter'), '9000');
function command(name, body) {
    const f = path.join(tmp, 'bin', name);
    fs.writeFileSync(f, '#!/bin/bash\nset -eu\n' + body + '\n');
    fs.chmodSync(f, 0o700);
}
command('sbatch', 'node "$MOCK_ROOT/sbatch.js" "$@"');
command('squeue', 'cat "$MOCK_ROOT/queue"');
command('sinfo', 'printf "gpu:l40s:4\\n"');
command('scontrol', 'if [ "$1" = show ]; then printf "ClusterName = %s_mock\\n" "$MOCK_SITE"; else printf "%s\\n" "$*" >> "$MOCK_ROOT/updates"; fi');
fs.writeFileSync(path.join(tmp, 'sbatch.js'), `
const fs=require('fs'),p=process.env.MOCK_ROOT;
const id=Number(fs.readFileSync(p+'/counter','utf8'))+1;
fs.writeFileSync(p+'/counter',String(id));
fs.appendFileSync(p+'/calls',JSON.stringify({id,args:process.argv.slice(2)})+'\\n');
console.log(id);
`);
const env = {...process.env, MOCK_ROOT: tmp, SOURCE_ROOT: path.join(tmp, 'project'),
    ENV_ACTIVATE: path.join(tmp, 'activate'), DATA_ROOT: path.join(root, 'data'),
    PATH: path.join(tmp, 'bin') + ':' + process.env.PATH};
let site = 'killarney';
function run(expectSuccess = true) {
    const r = spawnSync('sh', [path.join(tmp, 'submit_basis_lambda_' + site + '_ghaddar.sh')],
        {env: {...env, MOCK_SITE: site, BASIS_SITE: site}, encoding: 'utf8'});
    if (expectSuccess) assert.equal(r.status, 0, r.stdout + r.stderr);
    return r;
}
function calls() { return fs.existsSync(path.join(tmp, 'calls')) ? fs.readFileSync(path.join(tmp, 'calls'), 'utf8').trim().split('\n').map(JSON.parse) : []; }
function queue(ids) { fs.writeFileSync(path.join(tmp, 'queue'), ids.map(i => i + ' PENDING\n').join('')); }
try {
    const resnet = path.join(pools, 'ResNet20BN_60ep_lr0.1_bs128_seed42');
    fs.renameSync(resnet, resnet + '.absent');
    const rejected = run(false);
    assert.notEqual(rejected.status, 0);
    assert(rejected.stderr.includes('missing surrogate:'));
    assert.equal(calls().length,0);
    fs.renameSync(resnet + '.absent', resnet);
    for (site of ['killarney', 'vulcan']) {
        const offset = calls().length, tasks = plan.tasks.filter(t => t.site === site);
        queue([]);
        run();
        const initial = calls().slice(offset), ids = new Map();
        assert.equal(initial.length, 33);
        for (let i = 0; i < initial.length; i++) {
            const task = tasks[i], call = initial[i];
            assert(task.seconds <= 21600);
            assert(call.args.includes('--account=aip-bghaddar'));
            assert(call.args.includes(site === 'vulcan' ? '--gres=gpu:l40s:1' : '--gres=gpu:1'));
            assert(call.args.includes('--cpus-per-task=1'));
            assert(call.args.includes('--mem=' + task.memory));
            assert(call.args.includes('--time=' + [Math.floor(task.seconds / 3600), Math.floor(task.seconds / 60) % 60, task.seconds % 60].map(n => String(n).padStart(2, '0')).join(':')));
            assert.equal(call.args.at(-1),site);
            const dep = call.args.find(a => a.startsWith('--dependency='));
            assert.equal(dep, task.requires.length ? '--dependency=afterok:' + task.requires.map(p => ids.get(p)).join(':') : undefined);
            ids.set(task.id, call.id);
        }
        queue(initial.map(c => c.id));
        run();
        assert.equal(calls().length, offset+33, 'Active tasks duplicated');
        const craft = tasks.find(t => t.stage === 'craft');
        const oldCraft = ids.get(craft.id);
        queue(initial.map(c => c.id).filter(i => i !== oldCraft));
        run();
        assert.equal(calls().length, offset+34);
        const replacement = calls().at(-1).id;
        const updates = fs.readFileSync(path.join(tmp, 'updates'), 'utf8').trim().split('\n');
        assert(updates.at(-1).includes(String(replacement)));
        const doneTask = tasks[0];
        const hash = crypto.createHash('sha256').update(fs.readFileSync(path.join(batch, 'plan.json')))
            .update(fs.readFileSync(path.join(batch, 'code.sha256'))).digest('hex');
        const done = path.join(tmp, 'project/rebuttal_basis_lambda_result/jobs', doneTask.id);
        fs.mkdirSync(done, {recursive: true});
        fs.writeFileSync(path.join(done, '.done'), JSON.stringify({protocol_hash: hash}));
        queue([...initial.map(c => c.id).filter(i => ![oldCraft, ids.get(doneTask.id)].includes(i)), replacement]);
        assert(run().stdout.includes('SKIP ' + doneTask.id + ': complete'));
        assert.equal(calls().length, offset+34);
    }
    const updates = fs.readFileSync(path.join(tmp,'updates'),'utf8').trim().split('\n');
    assert.equal(updates.length,2,'Each site repairs one local downstream job');
    assert(fs.existsSync(path.join(batch,'submissions_vulcan.tsv')));
    assert(fs.existsSync(path.join(batch,'submissions_killarney.tsv')));
    // Check node-local staging paths without copying data or starting Python.
    const realSha = spawnSync('which', ['sha256sum'], {encoding:'utf8'}).stdout.trim();
    fs.writeFileSync(path.join(tmp,'bash_env'), 'module() { :; }\n');
    command('sha256sum', 'if [ "$PWD" = "$MOCK_BATCH" ]; then exec "$REAL_SHA" "$@"; else cat >/dev/null; fi');
    command('rsync', '[ -s "$2" ]; printf "%s\\n" "$2" >> "$MOCK_ROOT/staged"');
    command('srun', 'printf "mocked worker\\n"');
    for (const site of ['vulcan','killarney']) {
        fs.writeFileSync(path.join(tmp,'staged'),'');
        const task = plan.tasks.find(t => t.site === site && t.dataset === 'TinyImageNet');
        const stageEnv = {...env, BASH_ENV:path.join(tmp,'bash_env'), MOCK_BATCH:batch,
            REAL_SHA:realSha, SLURM_TMPDIR:path.join(tmp,'node_'+site)};
        const r=spawnSync('bash',[path.join(batch,'job.sh'),env.SOURCE_ROOT,batch,
            task.id,env.ENV_ACTIVATE,env.DATA_ROOT,site],{env:stageEnv,encoding:'utf8'});
        assert.equal(r.status,0,r.stdout+r.stderr);
        const staged=fs.readFileSync(path.join(tmp,'staged'),'utf8').trim().split('\n');
        const models=staged.filter(f=>f.includes('/cache/surrogates/'));
        assert.equal(models.length,20);
        assert(models.every(f=>f.includes('seed42_new/') === (site === 'killarney')));
    }
    console.log('PASS: disjoint 33+33 jobs, 264 trainings/site, exact settings, correct GPUs/input paths, local dependencies, separate ledgers and resume.');
} finally {
    fs.rmSync(tmp, {recursive: true, force: true});
}
