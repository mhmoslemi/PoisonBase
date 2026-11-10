const fs = require('fs'), path = require('path'), cp = require('child_process');
const assert = require('assert'), crypto = require('crypto');
const { dependencies } = require('./submit.js');

function main(root) {
  const batch = path.join(root, 'sbatch/rebuttal_tiny_gm_vgg13_5x5');
  const call = (name, args) => cp.execFileSync(name, args, { cwd: batch, encoding: 'utf8' }).trim();
  assert(/ClusterName\s*=\s*fir\b/i.test(call('scontrol', ['show', 'config'])), 'Run this on Fir');
  call('sha256sum', ['--check', '--quiet', 'code.sha256']);
  assert(/gpu:h100:/i.test(call('sinfo', ['-h', '-o', '%G', '--Node'])), 'No full H100 advertised');
  const bytes = fs.readFileSync(path.join(batch, 'plan.json')), plan = JSON.parse(bytes);
  const hash = crypto.createHash('sha256').update(bytes).update(fs.readFileSync(path.join(batch, 'code.sha256'))).digest('hex');
  const tasks = new Map(plan.tasks.map(t => [t.id, t]));
  assert.equal(fs.statSync(path.join(root, 'data/tinyimagenet.pt')).size, plan.data_bytes);
  assert(fs.existsSync('/home/mmoslem3/ENV/bin/activate'));
  const results = path.join(root, plan.results), finished = new Set();
  for (const task of plan.tasks) {
    const marker = path.join(results, 'jobs', task.id, '.done');
    if (!fs.existsSync(marker)) continue;
    const saved = JSON.parse(fs.readFileSync(marker));
    assert.equal(saved.protocol_hash, hash, 'Incompatible result ' + task.id);
    assert.deepEqual(saved.task, task);
    finished.add(task.id);
  }
  const queue = new Map(call('squeue', ['-h', '-u', 'mmoslem3', '-o', '%i|%T']).split('\n').filter(Boolean).map(line => line.split('|')));
  const source = new Map();
  const originalLedger = path.join(batch, 'submissions_fir.tsv');
  assert(fs.existsSync(originalLedger), 'Missing original Fir submission ledger');
  for (const line of fs.readFileSync(originalLedger, 'utf8').trim().split('\n')) {
    const [task, job] = line.split('\t');
    assert(tasks.has(task) && /^\d+$/.test(job));
    if (queue.has(job)) source.set(task, job);
  }
  const pairedSchedule = path.join(batch, 'fir_pairs.json'), pairedLedger = path.join(batch, 'pairs_submissions_fir.tsv');
  if (fs.existsSync(pairedSchedule) && fs.existsSync(pairedLedger)) {
    const pairs = JSON.parse(fs.readFileSync(pairedSchedule));
    assert.equal(pairs.protocol_hash, hash);
    const latest = new Map(fs.readFileSync(pairedLedger, 'utf8').trim().split('\n').map(line => line.split('\t').slice(0, 2)));
    for (const group of pairs.groups) if (queue.has(latest.get(group.id)))
      for (const id of group.tasks) source.set(id, latest.get(group.id));
    for (const [id, job] of pairs.external) if (queue.has(job)) source.set(id, job);
  }
  for (const task of plan.tasks) if (task.stage !== 'surrogate' && !finished.has(task.id))
    assert(!source.has(task.id), 'Older crafting/victim job still active: ' + source.get(task.id) + '. Cancel its pending job before replacing it.');
  const locations = new Map();
  for (const task of plan.tasks.filter(t => t.stage === 'surrogate')) {
    if (finished.has(task.id)) {
      const index = task.surrogates[0], model = path.join(results, 'surrogates', 'net_' + index + '.pt');
      assert(fs.statSync(model).size > 0, 'Missing completed model ' + model);
      const metadata = JSON.parse(fs.readFileSync(model.replace(/\.pt$/, '.json')));
      assert.equal(metadata.epochs, 150);
      assert.equal(metadata.identity.protocol_hash, hash);
      assert.equal(metadata.identity.victim, index);
      assert.equal(metadata.seed, plan.cells[0].args.seed * 1000 + index);
    } else {
      assert(source.has(task.id), 'Surrogate is unfinished and has no queued/running job: ' + task.id);
      assert(['PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'].includes(queue.get(source.get(task.id))), 'Unhealthy surrogate job ' + source.get(task.id));
      locations.set(task.id, source.get(task.id));
    }
  }
  console.log('Waiting for surrogate jobs:', [...locations].map(([id, job]) => id + '=' + job).join(', ') || 'none; all 20 complete');
  // Keep logs and scheduling records outside the checkout that sync replaces.
  const runtime = path.join(path.dirname(root), 'tiny_gm_vgg13_fir_runtime');
  fs.mkdirSync(path.join(runtime, 'logs'), { recursive: true });
  const scheduleFile = path.join(runtime, 'finish_schedule.json');
  let schedule;
  if (fs.existsSync(scheduleFile)) {
    schedule = JSON.parse(fs.readFileSync(scheduleFile));
    assert.equal(schedule.protocol_hash, hash);
    assert.equal(schedule.root, root);
  } else {
    const groups = [];
    for (const [stage, count] of [['craft', 3], ['victim', 9]]) {
      const remaining = plan.tasks.filter(t => t.stage === stage && !finished.has(t.id));
      for (let i = 0; i < remaining.length; i += count)
        groups.push({ id: stage + '_' + String(i / count).padStart(3, '0'), stage, tasks: remaining.slice(i, i + count).map(t => t.id) });
    }
    schedule = { protocol_hash: hash, root, groups };
    fs.writeFileSync(scheduleFile, JSON.stringify(schedule, null, 2) + '\n');
  }
  const ledger = path.join(runtime, 'finish_submissions.tsv'), latest = new Map();
  if (fs.existsSync(ledger)) for (const line of fs.readFileSync(ledger, 'utf8').trim().split('\n')) {
    const [group, job, dep] = line.split('\t');
    latest.set(group, { job, dep });
  }
  let submitted = 0;
  for (const group of schedule.groups) {
    const missing = group.tasks.filter(id => !finished.has(id));
    if (!missing.length) continue;
    const deps = dependencies([...new Set(missing.flatMap(id => tasks.get(id).requires))], finished, locations);
    const dependency = deps.length ? 'afterok:' + deps.join(':') : '';
    const previous = latest.get(group.id);
    if (previous && queue.has(previous.job)) {
      if (queue.get(previous.job) === 'PENDING' && (previous.dep === '-' ? '' : previous.dep) !== dependency) {
        call('scontrol', ['update', 'JobId=' + previous.job, 'Dependency=' + dependency]);
        fs.appendFileSync(ledger, [group.id, previous.job, dependency || '-', new Date().toISOString()].join('\t') + '\n');
      }
      for (const id of missing) locations.set(id, previous.job);
      console.log('ACTIVE', group.id, previous.job);
      continue;
    }
    const seconds = group.seconds ?? 11700;
    assert(Number.isSafeInteger(seconds) && seconds >= 120, 'Invalid time limit: ' + group.id);
    const time = [Math.floor(seconds / 3600), Math.floor(seconds % 3600 / 60), seconds % 60]
      .map(n => String(n).padStart(2, '0')).join(':');
    const reserve = seconds < 1800 ? 60 : 180;
    const args = ['--parsable', '--account=def-bghaddar', '--nodes=1', '--ntasks=1',
      '--cpus-per-task=1', '--mem=12G', '--gpus-per-node=h100:1', '--time=' + time,
      '--signal=USR1@' + reserve, '--job-name=tgv13f_' + group.id, '--chdir=' + root,
      '--output=' + path.join(runtime, 'logs/%x-%j.out')];
    if (dependency) args.push('--dependency=' + dependency);
    const job = call('sbatch', [...args, path.join(root, 'experiments/rebuttal_tiny_gm/paired_job.sh'), root, batch, String(seconds), ...missing]).split(';')[0];
    assert(/^\d+$/.test(job));
    fs.appendFileSync(ledger, [group.id, job, dependency || '-', new Date().toISOString()].join('\t') + '\n');
    for (const id of missing) locations.set(id, job);
    submitted++;
    console.log('SUBMITTED', group.id, job, missing.length, 'tasks');
  }
  console.log('Submitted', submitted, 'jobs; full schedule:', schedule.groups.filter(g => g.stage === 'craft').length,
    'crafting +', schedule.groups.filter(g => g.stage === 'victim').length, 'victim jobs. Time limits follow the saved schedule; all 12G, def-bghaddar, 1 CPU, 1 H100.');
  console.log('Logs:', path.join(runtime, 'logs'));
}

module.exports = { main };
if (require.main === module) main(process.argv[2]);
