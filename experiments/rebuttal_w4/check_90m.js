const fs = require('fs'), os = require('os'), path = require('path');
const assert = require('assert'), cp = require('child_process');
const root = path.resolve(__dirname, '../..'), batch = root + '/sbatch/rebuttal_w4_90m';
const raw = fs.readFileSync(batch + '/plan.json', 'utf8');
const originalRaw = fs.readFileSync(root + '/sbatch/rebuttal_w4/plan.json', 'utf8');
const plan = JSON.parse(raw), original = JSON.parse(originalRaw);
assert.equal(raw.slice(0, raw.indexOf('"tasks":')), originalRaw.slice(0, originalRaw.indexOf('"tasks":')));
assert.deepEqual(plan.protocol, original.protocol);
assert.deepEqual(plan.cells, original.cells);
assert.equal(plan.tasks.length, 20);
const seen = new Set();
for (const task of plan.tasks) {
  assert.equal(task.time, '01:30:00');
  const parent = original.tasks.find(t => t.id === task.parent_task);
  assert(parent && task.cell === parent.cell && task.method === parent.method);
  const cell = plan.cells.find(c => c.id === task.cell);
  assert(cell.model === 'ConvNetBN' ? task.units.length === 3 : task.units.length === 1);
  for (const unit of task.units) {
    assert(parent.units.some(u => u.draw === unit.draw && u.target === unit.target));
    const key = [task.cell, task.method, unit.draw, unit.target].join('|');
    assert(!seen.has(key)); seen.add(key);
  }
}
assert.equal(seen.size, 22);
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'w4-90m-check-'));
try {
  const localBatch = fixture + '/sbatch/rebuttal_w4_90m', bin = fixture + '/bin';
  fs.mkdirSync(localBatch, {recursive: true}); fs.mkdirSync(bin);
  for (const name of ['plan.json', 'job.sh', 'split_metadata.json', 'inputs.sha256'])
    fs.copyFileSync(batch + '/' + name, localBatch + '/' + name);
  fs.symlinkSync(root + '/sbatch/rebuttal_w4', fixture + '/sbatch/rebuttal_w4');
  fs.copyFileSync(root + '/submit_w4_90m_ghaddar.sh', fixture + '/submit_w4_90m_ghaddar.sh');
  fs.writeFileSync(fixture + '/activate', '');
  fs.writeFileSync(fixture + '/bash_env', 'module() { :; }\n');
  fs.writeFileSync(bin + '/squeue', '#!/bin/sh\nprintf "%s\\n" "$MOCK_ACTIVE"\n', {mode: 0o755});
  fs.writeFileSync(bin + '/sbatch', '#!' + process.execPath + '\n' +
    'const fs=require("fs"),p=process.env.MOCK_CALLS;' +
    'fs.appendFileSync(p,JSON.stringify(process.argv.slice(2))+"\\n");' +
    'console.log(880000+fs.readFileSync(p,"utf8").trim().split("\\n").length);\n', {mode: 0o755});
  const callsFile = fixture + '/calls'; fs.writeFileSync(callsFile, '');
  const env = {...process.env, PATH: bin + ':' + process.env.PATH, BASH_ENV: fixture + '/bash_env',
    MOCK_CALLS: callsFile, MOCK_ACTIVE: '', ENV_ACTIVATE: fixture + '/activate', W4_ACCOUNT: 'aip-bghaddar'};
  const run = (...args) => cp.spawnSync('sh', [fixture + '/submit_w4_90m_ghaddar.sh', ...args], {env, encoding: 'utf8'});
  const calls = () => fs.readFileSync(callsFile, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
  let r = run(); assert.equal(r.status, 0, r.stderr); assert.equal(calls().length, 20);
  for (const args of calls()) for (const flag of ['--account=aip-bghaddar', '--time=01:30:00',
    '--cpus-per-task=1', '--mem=6G', '--gres=gpu:l40s:1']) assert(args.includes(flag));
  const ledger = fs.readFileSync(localBatch + '/submissions.tsv', 'utf8').trim().split('\n').map(l => l.split('\t'));
  env.MOCK_ACTIVE = ledger.map(r => r[1]).join('\n');
  r = run(); assert.equal(r.status, 0, r.stderr); assert.equal(calls().length, 20);
  r = run('first_res_gm_b002'); assert.equal(r.status, 0, r.stderr); assert.equal(calls().length, 20);
  env.MOCK_ACTIVE = ledger.slice(1).map(r => r[1]).join('\n');
  r = run(ledger[0][0]); assert.equal(r.status, 0, r.stderr); assert.equal(calls().length, 21);
  env.MOCK_ACTIVE = '2206878'; r = run(); assert.notEqual(r.status, 0); assert.equal(calls().length, 21);
  r = run('unknown'); assert.notEqual(r.status, 0); assert.equal(calls().length, 21);
  console.log('PASS: same scientific protocol, 22 disjoint target/draw units, 20 jobs; correct resources; retries skip active jobs; unfinished jobs resume; overlapping parents are blocked.');
} finally { fs.rmSync(fixture, {recursive: true, force: true}); }
