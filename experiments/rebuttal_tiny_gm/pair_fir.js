const fs = require('fs'), path = require('path'), cp = require('child_process');
const assert = require('assert'), crypto = require('crypto');
const { dependencies } = require('./submit.js');

function main(root, action) {
  assert(['cancel', 'submit'].includes(action));
  assert(!fs.existsSync(path.join(path.dirname(root), 'tiny_gm_vgg13_fir_runtime/finish_schedule.json')),
    'Fir downstream now uses nine-victim jobs: run sh submit_tiny_finish_fir_ghaddar.sh');
  const batch = path.join(root, 'sbatch/rebuttal_tiny_gm_vgg13_5x5');
  const call = (name, args) => cp.execFileSync(name, args, { cwd: batch, encoding: 'utf8' }).trim();
  assert(/ClusterName\s*=\s*fir\b/i.test(call('scontrol', ['show', 'config'])), 'Run this on Fir');
  call('sha256sum', ['--check', '--quiet', 'code.sha256']);
  const bytes = fs.readFileSync(path.join(batch, 'plan.json')), plan = JSON.parse(bytes);
  const hash = crypto.createHash('sha256').update(bytes).update(fs.readFileSync(path.join(batch, 'code.sha256'))).digest('hex');
  const tasks = new Map(plan.tasks.map(t => [t.id, t]));
  const results = path.join(root, plan.results);
  const sourceLedger = path.join(batch, 'submissions_fir.tsv');
  assert(fs.existsSync(sourceLedger), 'Missing original Fir submission ledger');
  const source = new Map();
  for (const line of fs.readFileSync(sourceLedger, 'utf8').trim().split('\n')) {
    const [task, job] = line.split('\t');
    assert(tasks.has(task) && /^\d+$/.test(job), 'Invalid Fir submission ledger');
    source.set(task, job);
  }
  assert.equal(source.size, 338, 'Expected the complete original Fir ledger');
  const protectedJobs = new Set(['3197751', '3197752', '3197753']);
  for (let i = 0; i < 3; i++) assert(protectedJobs.has(source.get('surrogate_0' + i)), 'Protected surrogate job differs');
  const queue = new Map(call('squeue', ['-h', '-u', 'mmoslem3', '-o', '%i|%T']).split('\n').filter(Boolean).map(line => line.split('|')));
  const finished = new Set();
  for (const task of plan.tasks) {
    const marker = path.join(results, 'jobs', task.id, '.done');
    if (!fs.existsSync(marker)) continue;
    const saved = JSON.parse(fs.readFileSync(marker));
    assert.equal(saved.protocol_hash, hash, 'Incompatible completed task ' + task.id);
    assert.deepEqual(saved.task, task);
    finished.add(task.id);
  }
  const pending = [...new Set([...source.values()].filter(job => queue.get(job) === 'PENDING' && !protectedJobs.has(job)))];
  if (action === 'cancel') {
    if (pending.length) call('scancel', ['--state=PENDING', ...pending]);
    console.log('Canceled', pending.length, 'pending jobs from this experiment. Running jobs were left alone.');
    return;
  }
  assert.equal(pending.length, 0, 'Run sh cancel_tiny_pending_fir.sh first');
  assert(/gpu:h100:/i.test(call('sinfo', ['-h', '-o', '%G', '--Node'])), 'No full H100 GPU advertised');
  assert.equal(fs.statSync(path.join(root, 'data/tinyimagenet.pt')).size, plan.data_bytes);
  assert(fs.existsSync('/home/mmoslem3/ENV/bin/activate'));
  const scheduleFile = path.join(batch, 'fir_pairs.json');
  let schedule;
  if (fs.existsSync(scheduleFile)) {
    schedule = JSON.parse(fs.readFileSync(scheduleFile));
    assert.equal(schedule.protocol_hash, hash);
    assert.equal(schedule.root, root);
  } else {
    const external = [...source].filter(([id, job]) => !finished.has(id) && queue.has(job));
    for (const [id, job] of source) if (protectedJobs.has(job) && !finished.has(id))
      assert(queue.has(job), 'Protected job ' + job + ' stopped without a completed result; inspect it first');
    const kept = new Set(external.map(([id]) => id)), groups = [];
    for (const stage of ['surrogate', 'craft', 'victim']) {
      const remaining = plan.tasks.filter(t => t.stage === stage && !finished.has(t.id) && !kept.has(t.id));
      for (let i = 0; i < remaining.length; i += 2)
        groups.push({ id: stage + '_' + String(i / 2).padStart(3, '0'), stage, tasks: remaining.slice(i, i + 2).map(t => t.id) });
    }
    schedule = { protocol_hash: hash, root, external, groups };
    fs.writeFileSync(scheduleFile, JSON.stringify(schedule, null, 2) + '\n');
  }
  const ledger = path.join(batch, 'pairs_submissions_fir.tsv'), latest = new Map();
  if (fs.existsSync(ledger)) for (const line of fs.readFileSync(ledger, 'utf8').trim().split('\n')) {
    const [group, job, dep] = line.split('\t');
    latest.set(group, { job, dep });
  }
  const locations = new Map();
  for (const [id, job] of schedule.external) if (!finished.has(id)) {
    assert(queue.has(job), 'Preserved job ' + job + ' stopped without completing; inspect it first');
    locations.set(id, job);
  }
  const logs = path.join(batch, 'logs/fir_pairs');
  fs.mkdirSync(logs, { recursive: true });
  let submitted = 0;
  for (const group of schedule.groups) {
    const missing = group.tasks.filter(id => !finished.has(id));
    if (!missing.length) continue;
    const required = [...new Set(missing.flatMap(id => tasks.get(id).requires))];
    const deps = dependencies(required, finished, locations);
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
    const seconds = 7200;
    const time = [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60].map(n => String(n).padStart(2, '0')).join(':');
    const args = ['--parsable', '--account=def-bghaddar', '--nodes=1', '--ntasks=1',
      '--cpus-per-task=1', '--mem=11G', '--gpus-per-node=h100:1', '--time=' + time,
      '--signal=USR1@180', '--job-name=tgv13p_' + group.id, '--chdir=' + root,
      '--output=' + path.join(logs, '%x-%j.out')];
    if (dependency) args.push('--dependency=' + dependency);
    const job = call('sbatch', [...args, path.join(root, 'experiments/rebuttal_tiny_gm/paired_job.sh'), root, batch, String(seconds), ...missing]).split(';')[0];
    assert(/^\d+$/.test(job));
    fs.appendFileSync(ledger, [group.id, job, dependency || '-', new Date().toISOString()].join('\t') + '\n');
    for (const id of missing) locations.set(id, job);
    submitted++;
    console.log('SUBMITTED', group.id, job, time, missing.join(' + '));
  }
  console.log('Submitted', submitted, 'paired jobs, def-bghaddar, 11G, 1 CPU, 1 H100. Rerun the same script to resume.');
}

module.exports = { main };
if (require.main === module) main(process.argv[2], process.argv[3]);
