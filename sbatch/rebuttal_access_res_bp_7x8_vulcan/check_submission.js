// Lightweight checks only: no Slurm submissions, Python, or GPU work.
const fs = require('fs');
const path = require('path');
const os = require('os');
const assert = require('assert');
const {spawnSync} = require('child_process');
const root = path.resolve(__dirname, '../..');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'w1-res-bp-check-'));
try {
    const batch = path.join(tmp, 'sbatch/rebuttal_access_res_bp_7x8_vulcan');
    fs.mkdirSync(path.dirname(batch), {recursive: true});
    fs.cpSync(__dirname, batch, {recursive: true,
        filter: p => !['logs', 'submission.lock', 'submissions.tsv'].includes(path.basename(p))});
    fs.copyFileSync(path.join(root, 'submit_w1_resnet_bp_7x8_vulcan.sh'), path.join(tmp, 'submit.sh'));
    for (const dir of ['cache', 'data', 'target_sets', 'reviewer_20261106_fir'])
        fs.symlinkSync(path.join(root, dir), path.join(tmp, dir));
    fs.mkdirSync(path.join(tmp, 'bin'));
    fs.writeFileSync(path.join(tmp, 'activate'), '# mocked environment\n');
    fs.writeFileSync(path.join(tmp, 'queue'), '');
    fs.writeFileSync(path.join(tmp, 'counter'), '500000');
    function command(name, body) {
        const file = path.join(tmp, 'bin', name);
        fs.writeFileSync(file, '#!/bin/bash\nset -eu\n' + body + '\n');
        fs.chmodSync(file, 0o700);
    }
    command('sinfo', 'printf "gpu:l40s:4\\n"');
    command('squeue', 'cat "$MOCK_ROOT/queue"');
    command('sbatch', 'printf "%s\\n" "$*" >> "$MOCK_ROOT/calls"; n=$(cat "$MOCK_ROOT/counter"); n=$((n+1)); printf "%s" "$n" > "$MOCK_ROOT/counter"; printf "%s\\n" "$n"');
    const env = {...process.env, MOCK_ROOT: tmp, ENV_ACTIVATE: path.join(tmp, 'activate'),
        PATH: path.join(tmp, 'bin') + ':' + process.env.PATH};
    const run = () => {
        const r = spawnSync('sh', [path.join(tmp, 'submit.sh')], {env, encoding: 'utf8'});
        assert.equal(r.status, 0, r.stdout + r.stderr);
        return r.stdout;
    };
    const manifest = fs.readFileSync(path.join(batch, 'manifest.tsv'), 'utf8').trim().split('\n')
        .slice(1).map(l => l.split('\t'));
    assert.equal(manifest.length, 41);
    assert.equal(manifest.filter(t => t[2] === 'random').length, 6);
    assert.equal(manifest.filter(t => t[2] === 'basis').length, 35);
    const coverage = new Map();
    for (const [id, attack, method, condition, start, stop] of manifest) {
        assert.equal(attack, 'fc');
        assert(+stop > +start && +stop - +start <= 2);
        const key = method + ':' + condition;
        if (!coverage.has(key)) coverage.set(key, new Set());
        for (let draw = +start; draw < +stop; draw++) {
            assert(!coverage.get(key).has(draw), 'Duplicate draw across jobs');
            coverage.get(key).add(draw);
        }
    }
    assert.equal(coverage.size, 10);
    let trainings = 0;
    for (const [key, draws] of coverage) {
        const count = key.startsWith('random:') ? 3 : 9;
        assert.deepEqual([...draws].sort((a,b)=>a-b), Array.from({length:count}, (_,i)=>i));
        trainings += count * 7 * 8;
    }
    assert.equal(trainings, 4032);
    run();
    let calls = fs.readFileSync(path.join(tmp, 'calls'), 'utf8').trim().split('\n');
    assert.equal(calls.length, 41);
    for (const call of calls) for (const argument of ['--account=aip-bghaddar', '--time=04:00:00',
        '--cpus-per-task=1', '--mem=6G', '--gres=gpu:l40s:1']) assert(call.split(' ').includes(argument));
    fs.writeFileSync(path.join(tmp, 'queue'), manifest.map((_,i)=>500001+i).join('\n')+'\n');
    run();
    assert.equal(fs.readFileSync(path.join(tmp, 'calls'), 'utf8').trim().split('\n').length, 41);
    const completed = path.join(tmp, 'rebuttal_access_res_bp_7x8_result/jobs/0');
    fs.mkdirSync(completed, {recursive:true});
    fs.writeFileSync(path.join(completed, '.done'), 'complete\n');
    fs.writeFileSync(path.join(tmp, 'queue'), manifest.slice(2).map((_,i)=>500003+i).join('\n')+'\n');
    assert(run().includes('SKIP task 0: complete'));
    calls = fs.readFileSync(path.join(tmp, 'calls'), 'utf8').trim().split('\n');
    assert.equal(calls.length, 42, 'Only the unfinished inactive job should resume');
    const table = fs.readFileSync(path.join(root, 'Rebutal/Rebutal.tex'), 'utf8')
        .split('\\label{tab:reb-access}')[1].split('\\end{table}')[0];
    const rows = table.split('\n').filter(l=>/^& (100|50 |20 )/.test(l));
    assert.equal(rows.length, 7);
    assert(rows.every(l=>l.split('&').length===8));
    assert.equal((rows.join('\n').match(/\\tbd/g)||[]).length,10);
    const basis = rows.map(l=>l.split('&')[3].trim());
    assert.deepEqual(basis,['66.9','49.8','30.8','53.2','48.0','39.3','16.3']);
    console.log('PASS: 41 jobs; ten ResNet20 BP cells; 4,032 trials; exact resources; active/done skip and resume; table columns and existing ConvNet values.');
} finally {
    fs.rmSync(tmp, {recursive:true, force:true});
}
