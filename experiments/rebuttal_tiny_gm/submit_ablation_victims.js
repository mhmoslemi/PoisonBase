const fs = require('fs'), path = require('path'), cp = require('child_process');
const assert = require('assert'), crypto = require('crypto');

const root = process.argv[2], batch = path.join(root, 'sbatch/rebuttal_tiny_gm_vgg13_5x5');
const runtime = path.join(path.dirname(root), 'tiny_gm_vgg13_vulcan_runtime');
assert(!fs.existsSync(path.join(path.dirname(root), 'tiny_gm_vgg13_killarney_runtime/migration.json')),
  'These jobs are managed by tiny_gm_vgg13_killarney_runtime/submit.js on Killarney.');
const call = (name, args) => cp.execFileSync(name, args, {cwd: batch, encoding: 'utf8'}).trim();
assert(/ClusterName\s*=\s*vulcan\b/i.test(call('scontrol', ['show', 'config'])));
assert(/gpu:l40s:/i.test(call('sinfo', ['-h', '-o', '%G', '--Node'])));
assert(call('sacctmgr', ['show', 'assoc', 'user=mmoslem3', 'format=Account', '-p']).includes('aip-bghaddar|'));
call('sha256sum', ['--check', '--quiet', 'code.sha256']);
const bytes = fs.readFileSync(path.join(batch, 'plan.json')), plan = JSON.parse(bytes);
const hash = crypto.createHash('sha256').update(bytes).update(fs.readFileSync(path.join(batch, 'code.sha256'))).digest('hex');
const source = plan.cells.find(c => c.id === 'alligator_tarantula_b0005_basis');
assert.equal(source.targets.length, 5); assert.equal(source.args.num_victims, 5);
assert.equal(source.args.victim_epochs, 120);
const results = path.join(root, plan.results), ledger = path.join(runtime, 'ablation_victims.json');
const schedule = fs.existsSync(ledger) ? JSON.parse(fs.readFileSync(ledger)) : {source_protocol_hash: hash, records: []};
assert.equal(schedule.source_protocol_hash, hash);
const queue = new Map(call('squeue', ['-h', '-u', 'mmoslem3', '-o', '%i|%T']).split('\n').filter(Boolean).map(l => l.split('|')));
const canonical = JSON.parse(fs.readFileSync(path.join(runtime, 'moved_craft_from_fir_3225878.json')));
let reused = 0, existing = 0;
for (const target of source.targets) for (let v = 0; v < 5; v++) {
  const file = path.join(results, 'cells', source.id, 'target_' + target, 'victim_' + v + '.json');
  if (fs.existsSync(file)) {
    const r = JSON.parse(fs.readFileSync(file));
    assert.deepStrictEqual(r.identity, {protocol_hash: hash, cell: source.id, target, victim: v});
    assert.equal(r.epochs, 120); reused++;
  } else {
    const r = canonical.records.find(r => r.task.cell === source.id && r.target === target && r.task.victim === v);
    assert(r && ['PENDING', 'RUNNING', 'CONFIGURING'].includes(queue.get(r.vulcan_job)), 'Unscheduled lambda=1 victim');
    existing++;
  }
}
const specs = [['M', 'M', 1], ['R', 'R', 1], ['basis_lam0p1', 'BASIS', 0.1], ['basis_lam10', 'BASIS', 10]];
for (const [variant, component, weight] of specs) {
  const cell = structuredClone(source); cell.id = 'alligator_tarantula_b0005_' + variant;
  cell.component = component; cell.args.lambda_margin = weight;
  const folder = path.join(results, 'cells', cell.id), config = JSON.parse(fs.readFileSync(path.join(folder, 'craft_interactive.json')));
  assert.equal(config.source_protocol_hash, hash); assert(/^[0-9a-f]{64}$/.test(config.protocol_hash));
  assert.deepStrictEqual(config.cell, cell, 'Crafting settings changed: ' + variant);
  for (const target of cell.targets) {
    const dir = path.join(folder, 'target_' + target);
    assert(fs.statSync(path.join(dir, 'poison.pt')).size > 0, 'Missing saved poison');
    const base = JSON.parse(fs.readFileSync(path.join(dir, 'bases.json')));
    assert.deepStrictEqual(base.identity, {protocol_hash: config.protocol_hash, cell: cell.id, target});
    assert.equal(base.indices.length, 50); assert.equal(new Set(base.indices).size, 50);
    for (let v = 0; v < 5; v++) {
      const id = variant + '_t' + target + '_v' + v;
      const saved = schedule.records.find(r => r.id === id);
      const record = {id, variant, cell: cell.id, target, victim: v, protocol_hash: config.protocol_hash};
      if (saved) for (const [key, value] of Object.entries(record)) assert.deepStrictEqual(saved[key], value);
      else schedule.records.push(record);
    }
  }
}
assert.equal(schedule.records.length, 100); assert.equal(new Set(schedule.records.map(r => r.id)).size, 100);
function save() { fs.writeFileSync(ledger + '.tmp', JSON.stringify(schedule, null, 2) + '\n'); fs.renameSync(ledger + '.tmp', ledger); }
save();
let submitted = 0, done = 0, active = 0;
for (const r of schedule.records) {
  const final = path.join(results, 'cells', r.cell, 'target_' + r.target, 'victim_' + r.victim + '.json');
  if (fs.existsSync(final)) {
    const x = JSON.parse(fs.readFileSync(final));
    assert.deepStrictEqual(x.identity, {protocol_hash: r.protocol_hash, cell: r.cell, target: r.target, victim: r.victim});
    assert.equal(x.epochs, 120); assert.equal(x.model, 'VGG13BN'); done++; continue;
  }
  if (r.job && ['PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'].includes(queue.get(r.job))) { active++; continue; }
  const args = ['--parsable', '--account=aip-bghaddar', '--nodes=1', '--ntasks=1', '--cpus-per-task=1',
    '--mem=12G', '--gres=gpu:l40s:1', '--time=00:45:00', '--signal=USR1@180', '--chdir=' + root,
    '--job-name=tgv13_ab_' + r.id, '--output=' + path.join(runtime, 'logs/%x-%j.out'),
    path.join(root, 'experiments/rebuttal_tiny_gm/interactive_craft.sh'), r.variant,
    '--target', String(r.target), '--victim', String(r.victim)];
  r.job = call('sbatch', args).split(';')[0]; assert(/^\d+$/.test(r.job));
  r.account = 'aip-bghaddar'; r.seconds = 2700; r.submitted = new Date().toISOString(); save();
  submitted++; console.log('SUBMITTED', r.job, r.id);
}
console.log(JSON.stringify({submitted, done, active,lambda1_completed_reused: reused, lambda1_already_scheduled: existing,
  resources: 'aip-bghaddar, 45min, 12G, 1 CPU, 1 L40S', ledger}));
