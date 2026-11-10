const fs = require('fs'), path = require('path'), cp = require('child_process');
const assert = require('assert'), crypto = require('crypto');

function main(root) {
  const batch = path.join(root, 'sbatch/rebuttal_tiny_gm_vgg13_5x5');
  const runtime = path.join(path.dirname(root), 'tiny_gm_vgg13_vulcan_runtime');
  assert(!fs.existsSync(path.join(path.dirname(root), 'tiny_gm_vgg13_killarney_runtime/migration.json')),
    'These jobs are managed by tiny_gm_vgg13_killarney_runtime/submit.js on Killarney.');
  const manifestFile = path.join(runtime, 'moved_craft_from_fir_3225878.json');
  const manifest = JSON.parse(fs.readFileSync(manifestFile));
  assert.equal(manifest.root, root);
  assert(manifest.fir_cancelled, 'Cancel the recorded Fir source jobs before submitting on Vulcan');
  const call = (name, args) => cp.execFileSync(name, args, { cwd: batch, encoding: 'utf8' }).trim();
  assert(/ClusterName\s*=\s*vulcan\b/i.test(call('scontrol', ['show', 'config'])));
  assert(/gpu:l40s:/i.test(call('sinfo', ['-h', '-o', '%G', '--Node'])));
  const accounts = call('sacctmgr', ['show', 'assoc', 'user=mmoslem3', 'format=Account', '-p']);
  for (const r of manifest.records) {
    const account = r.account || 'aip-bghaddar';
    assert(['aip-bghaddar', 'aip-boyuwang', 'aip-yiweilu'].includes(account));
    assert(accounts.includes(account + '|'), 'Unavailable account: ' + account);
  }
  call('sha256sum', ['--check', '--quiet', 'code.sha256']);
  const bytes = fs.readFileSync(path.join(batch, 'plan.json'));
  const plan = JSON.parse(bytes), tasks = new Map(plan.tasks.map(t => [t.id, t]));
  const hash = crypto.createHash('sha256').update(bytes).update(fs.readFileSync(path.join(batch, 'code.sha256'))).digest('hex');
  assert.equal(hash, manifest.protocol_hash);
  assert.equal(fs.statSync(path.join(root, 'data/tinyimagenet.pt')).size, plan.data_bytes);
  assert(fs.existsSync('/home/mmoslem3/ENV/bin/activate'));
  const results = path.join(root, plan.results);
  const ready = JSON.parse(fs.readFileSync(path.join(results, 'preflight.json')));
  assert(ready.complete && ready.protocol_hash === hash);
  for (let i = 0; i < 20; i++) {
    const model = path.join(results, 'surrogates/net_' + i + '.pt');
    assert(fs.statSync(model).size > 0);
    const metadata = JSON.parse(fs.readFileSync(model.replace(/\.pt$/, '.json')));
    assert.equal(metadata.epochs, 150);
    assert.deepEqual(metadata.identity, { protocol_hash: hash, target: 'surrogate', victim: i });
    assert.equal(metadata.seed, plan.cells[0].args.seed * 1000 + i);
  }
  const crafts = manifest.records.filter(r => r.stage === 'craft');
  const victims = manifest.records.filter(r => r.stage === 'victim');
  assert.equal(crafts.length, 45); assert.equal(victims.length, 225);
  assert.equal(new Set(manifest.records.map(r => r.id)).size, 270);
  const byTarget = new Map(crafts.map(r => [r.task.cell + '/' + r.target, r]));
  assert.equal(byTarget.size, 45);
  for (const r of manifest.records) {
    assert.deepEqual(r.task, tasks.get(r.task.id), 'Scientific task changed: ' + r.id);
    assert.equal(r.stage, r.task.stage);
    assert(r.task.targets.includes(r.target));
    assert.equal(r.seconds, r.stage === 'victim' ? 2700 :
      (plan.cells.find(c => c.id === r.task.cell).args.budget === 0.001 ? 1500 : 900));
    if (r.stage === 'victim') {
      const parent = byTarget.get(r.task.cell + '/' + r.target);
      assert(parent && r.task.requires.includes(parent.task.id));
    }
  }
  for (const craft of crafts) assert.equal(victims.filter(v =>
    v.task.cell === craft.task.cell && v.target === craft.target).length, 5);
  const queue = new Map(call('squeue', ['-h', '-u', 'mmoslem3', '-o', '%i|%T'])
    .split('\n').filter(Boolean).map(line => line.split('|')));
  const ledger = path.join(runtime, 'craft_target_submissions.tsv');
  const nativeLedger = path.join(batch, 'submissions.tsv');
  const overridesFile = path.join(batch, 'account_overrides_vulcan.json');
  const overrides = fs.existsSync(overridesFile) ? JSON.parse(fs.readFileSync(overridesFile)) : {};
  for (const r of victims) overrides[r.task.id] = r.account || 'aip-bghaddar';
  fs.writeFileSync(overridesFile, JSON.stringify(overrides, null, 2) + '\n');
  const locations = new Map(), finished = new Set();
  const remember = (r, dep) => {
    const row = [r.id, r.vulcan_job, dep || '-', new Date().toISOString()].join('\t') + '\n';
    fs.appendFileSync(ledger, row);
    if (r.stage === 'victim') fs.appendFileSync(nativeLedger, row);
    r.dependency = dep;
    fs.writeFileSync(manifestFile, JSON.stringify(manifest, null, 2) + '\n');
  };
  let submitted = 0;
  for (const r of [...crafts, ...victims]) {
    const marker = path.join(results, 'jobs', r.id, '.done');
    if (fs.existsSync(marker)) {
      const expected = { protocol_hash: hash, task: r.task };
      if (r.stage === 'craft') expected.target = r.target;
      assert.deepEqual(JSON.parse(fs.readFileSync(marker)), expected);
      const savedFile = r.stage === 'craft' ? 'poison.pt' : 'victim_' + r.task.victim + '.json';
      assert(fs.existsSync(path.join(results, 'cells', r.task.cell, 'target_' + r.target, savedFile)));
      finished.add(r.id);
      continue;
    }
    if (r.site === 'fir') continue;
    const parent = r.stage === 'victim' ? byTarget.get(r.task.cell + '/' + r.target) : null;
    const dependency = parent && !finished.has(parent.id) ? 'afterok:' + locations.get(parent.id) : '';
    assert(!dependency.includes('undefined'));
    if (r.vulcan_job && queue.has(r.vulcan_job)) {
      if (queue.get(r.vulcan_job) === 'PENDING' && r.dependency !== dependency) {
        call('scontrol', ['update', 'JobId=' + r.vulcan_job, 'Dependency=' + dependency]);
        remember(r, dependency);
      }
      locations.set(r.id, r.vulcan_job);
      continue;
    }
    const time = [Math.floor(r.seconds / 3600), Math.floor(r.seconds % 3600 / 60), r.seconds % 60]
      .map(n => String(n).padStart(2, '0')).join(':');
    const args = ['--parsable', '--account=' + (r.account || 'aip-bghaddar'), '--nodes=1', '--ntasks=1',
      '--cpus-per-task=1', '--mem=12G', '--gres=gpu:l40s:1', '--time=' + time,
      '--signal=USR1@' + (r.stage === 'craft' ? 60 : 180), '--chdir=' + root,
      '--job-name=tgv13_' + r.id, '--output=' + path.join(runtime, 'logs/%x-%j.out')];
    if (dependency) args.push('--dependency=' + dependency);
    const runner = r.stage === 'craft' ? 'craft_target_job.sh' : 'paired_job.sh';
    args.push(path.join(root, 'experiments/rebuttal_tiny_gm', runner), root, batch,
      String(r.seconds), r.task.id);
    if (r.stage === 'craft') args.push(String(r.target));
    r.vulcan_job = call('sbatch', args).split(';')[0];
    assert(/^\d+$/.test(r.vulcan_job));
    r.submitted = new Date().toISOString();
    remember(r, dependency);
    locations.set(r.id, r.vulcan_job);
    submitted++;
    console.log('SUBMITTED', r.vulcan_job, r.stage, r.id, time, dependency);
  }
  console.log('Submitted:', submitted, 'Done:', finished.size, 'Active:', locations.size,
    'On Fir:', manifest.records.filter(r => r.site === 'fir').length);
}

if (require.main === module) main(process.argv[2]);
