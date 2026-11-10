const fs = require('fs');
const path = require('path');
const cp = require('child_process');
const crypto = require('crypto');
const assert = require('assert');

function dependencies(taskIds, finished, locations) {
  return [...new Set(taskIds.filter(id => !finished.has(id)).map(id => {
    assert(locations.has(id), 'Missing upstream task ' + id);
    return locations.get(id);
  }))];
}

function main(root, site = 'vulcan') {
  assert(['vulcan', 'killarney', 'fir'].includes(site), 'Unknown cluster ' + site);
  const remote = site !== 'vulcan';
  const account = site === 'fir' ? 'def-bghaddar' : 'aip-bghaddar';
  const gpu = site === 'fir' ? '--gpus-per-node=h100:1'
    : site === 'killarney' ? '--gres=gpu:1' : '--gres=gpu:l40s:1';
  const batch = path.join(root, 'sbatch/rebuttal_tiny_gm_vgg13_5x5');
  assert(site !== 'vulcan' || !fs.existsSync(path.join(path.dirname(root),
    'tiny_gm_vgg13_vulcan_runtime/moved_craft_from_fir_3225878.json')),
    'Split crafting now uses sh submit_tiny_targets_vulcan_ghaddar.sh');
  assert(site !== 'fir' || !fs.existsSync(path.join(path.dirname(root), 'tiny_gm_vgg13_fir_runtime/finish_schedule.json')),
    'Fir downstream now uses nine-victim jobs: run sh submit_tiny_finish_fir_ghaddar.sh');
  assert(site !== 'fir' || !fs.existsSync(path.join(batch, 'fir_pairs.json')),
    'Fir now uses paired jobs: run sh submit_tiny_pairs_fir_ghaddar.sh');
  const call = (name, args, options = {}) => cp.execFileSync(name, args,
    { encoding: 'utf8', cwd: batch, ...options }).trim();
  call('sha256sum', ['--check', '--quiet', 'code.sha256']);
  const planBytes = fs.readFileSync(path.join(batch, 'plan.json'));
  const plan = JSON.parse(planBytes);
  const hash = crypto.createHash('sha256').update(planBytes)
    .update(fs.readFileSync(path.join(batch, 'code.sha256'))).digest('hex');
  const results = path.join(root, plan.results);
  const tasks = new Map(plan.tasks.map(t => [t.id, t]));
  assert.equal(tasks.size, 338);
  const cluster = call('scontrol', ['show', 'config']);
  const clusterName = cluster.match(/ClusterName\s*=\s*(\S+)/)[1];
  assert(clusterName.toLowerCase().includes(site), 'Run the ' + site + ' script on ' + site);
  const inventory = call('sinfo', ['-h', '-o', '%G', '--Node']);
  assert(site === 'fir' ? /gpu:h100:/i.test(inventory) : /gpu:l40s?:/i.test(inventory),
    'Expected GPU type is unavailable on ' + site);
  const data = path.join(root, 'data/tinyimagenet.pt');
  if (remote && !fs.existsSync(data)) {
    const shared = path.join(process.env.DATA_ROOT || '/home/mmoslem3/scratch/data', 'tinyimagenet.pt');
    assert(fs.existsSync(shared), 'Copy data/tinyimagenet.pt to ' + data);
    fs.mkdirSync(path.dirname(data), { recursive: true });
    fs.symlinkSync(fs.realpathSync(shared), data);
  }
  assert.equal(fs.statSync(data).size, plan.data_bytes, 'Wrong Tiny ImageNet file: ' + data);
  assert(fs.existsSync('/home/mmoslem3/ENV/bin/activate'), 'Missing /home/mmoslem3/ENV/bin/activate');
  if (!remote) {
    const accounts = call('sacctmgr', ['show', 'assoc', 'user=mmoslem3', 'format=Account', '-p']);
    for (const code of ['aip-bghaddar', 'aip-yiweilu']) assert(accounts.includes(code + '|'));
  }
  const queue = call('squeue', ['-h', '-u', 'mmoslem3', '-o', '%i|%T']);
  const active = new Map();
  for (const line of queue.split('\n').filter(Boolean)) {
    const [job, state] = line.split('|');
    active.set(job, state);
  }
  const ledger = path.join(batch, remote ? 'submissions_' + site + '.tsv' : 'submissions.tsv');
  const logs = path.join(batch, 'logs', remote ? site : '');
  fs.mkdirSync(logs, { recursive: true });
  if (remote) {
    const identity = JSON.stringify({ cluster: clusterName, root, account });
    const identityFile = path.join(batch, 'cluster_' + site + '.txt');
    if (fs.existsSync(identityFile)) assert.equal(fs.readFileSync(identityFile, 'utf8'), identity,
      'Submission ledger belongs to another cluster/project');
    else fs.writeFileSync(identityFile, identity);
  }
  const latest = new Map();
  const history = [];
  if (fs.existsSync(ledger)) for (const line of fs.readFileSync(ledger, 'utf8').trim().split('\n')) {
    const [task, job, dep] = line.split('\t');
    latest.set(task, { job, dep });
    history.push({ task, job });
  }
  const overridesFile = path.join(batch, 'account_overrides_' + site + '.json');
  const overrides = fs.existsSync(overridesFile) ? JSON.parse(fs.readFileSync(overridesFile)) : {};
  let selected;
  if (process.env.TINY_REPLACE_OLD_JOBS) {
    assert.equal(site, 'fir');
    assert.equal(process.env.TINY_REPLACE_ACCOUNT, 'def-yiweilu');
    const oldJobs = new Set(process.env.TINY_REPLACE_OLD_JOBS.split(','));
    assert.equal(oldJobs.size, 4, 'Exactly four old job IDs are required');
    for (const job of oldJobs) {
      assert(/^\d+$/.test(job));
      assert(!active.has(job), 'Cancel job ' + job + ' before running this script');
      assert(history.some(row => row.job === job), 'Job ' + job + ' is missing from the Fir submission ledger');
    }
    selected = new Set(history.filter(row => oldJobs.has(row.job)).map(row => row.task));
    assert.equal(selected.size, 4);
    for (const id of selected) {
      assert.equal(tasks.get(id).stage, 'surrogate', 'Only the four surrogate jobs can be replaced');
      overrides[id] = process.env.TINY_REPLACE_ACCOUNT;
    }
    fs.writeFileSync(overridesFile, JSON.stringify(overrides, null, 2) + '\n');
  }
  const finished = new Set(), locations = new Map();
  for (const task of plan.tasks) {
    const marker = path.join(results, 'jobs', task.id, '.done');
    if (!fs.existsSync(marker)) continue;
    const saved = JSON.parse(fs.readFileSync(marker));
    assert.equal(saved.protocol_hash, hash, 'Wrong completed protocol ' + task.id);
    assert.deepEqual(saved.task, task);
    finished.add(task.id);
  }
  let submitted = 0;
  for (const task of plan.tasks) {
    if (finished.has(task.id)) continue;
    const deps = dependencies(task.requires, finished, locations);
    const dependency = deps.length ? 'afterok:' + deps.join(':') : '';
    const previous = latest.get(task.id);
    if (previous && active.has(previous.job)) {
      if (active.get(previous.job) === 'PENDING' && (previous.dep === '-' ? '' : previous.dep) !== dependency) {
        call('scontrol', ['update', 'JobId=' + previous.job, 'Dependency=' + dependency]);
        fs.appendFileSync(ledger, [task.id, previous.job, dependency || '-', new Date().toISOString()].join('\t') + '\n');
      }
      locations.set(task.id, previous.job);
      console.log('ACTIVE', task.id, previous.job);
      continue;
    }
    if (selected && !selected.has(task.id)) {
      if (previous) locations.set(task.id, previous.job);
      continue;
    }
    const selectedAccount = overrides[task.id] || (remote ? account : task.account);
    const args = ['--parsable', '--account=' + selectedAccount, '--nodes=1', '--ntasks=1',
      '--cpus-per-task=1', '--mem=12G', gpu, '--time=' + task.time,
      '--signal=USR1@180', '--job-name=tgv13_' + task.id, '--chdir=' + root,
      '--output=' + path.join(logs, '%x-%j.out')];
    if (dependency) args.push('--dependency=' + dependency);
    if (process.env.TINY_SUBMIT_TEST_ONLY === '1') {
      call('sbatch', ['--test-only', ...args, path.join(batch, 'job.sh'), root, batch, task.id]);
      console.log('SLURM TEST PASSED', task.id);
      return;
    }
    const job = call('sbatch', [...args, path.join(batch, 'job.sh'), root, batch, task.id]).split(';')[0];
    assert(/^\d+$/.test(job), 'Invalid job ID');
    fs.appendFileSync(ledger, [task.id, job, dependency || '-', new Date().toISOString()].join('\t') + '\n');
    locations.set(task.id, job);
    submitted++;
    console.log('SUBMITTED', task.id, job, selectedAccount, task.time);
  }
  console.log('TOTAL SUBMITTED:', submitted, 'individual GPU jobs. Already completed:', finished.size);
}

module.exports = { dependencies };
if (require.main === module) main(process.argv[2], process.argv[3]);
