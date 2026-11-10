const fs = require('fs'), path = require('path'), os = require('os');
const assert = require('assert'), cp = require('child_process'), vm = require('vm');
const root = path.resolve(__dirname, '../..');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'tiny-fir-pairs-'));
const relative = 'sbatch/rebuttal_tiny_gm_vgg13_5x5';
const batch = path.join(temporary, relative);
fs.mkdirSync(batch, { recursive: true });
try {
  for (const name of ['plan.json', 'code.sha256', 'job.sh', 'code', 'submissions_fir.tsv'])
    fs.cpSync(path.join(root, relative, name), path.join(batch, name), { recursive: true });
  fs.mkdirSync(path.join(temporary, 'data'));
  fs.symlinkSync(path.join(root, 'data/tinyimagenet.pt'), path.join(temporary, 'data/tinyimagenet.pt'));
  const plan = JSON.parse(fs.readFileSync(path.join(batch, 'plan.json')));
  const original = new Map(fs.readFileSync(path.join(batch, 'submissions_fir.tsv'), 'utf8').trim().split('\n').map(l => l.split('\t').slice(0, 2)));
  for (let i = 15; i <= 18; i++) {
    const id = 'surrogate_' + i, job = String(3199654 + i - 15);
    fs.appendFileSync(path.join(batch, 'submissions_fir.tsv'), [id, job, '-', 'test'].join('\t') + '\n');
    original.set(id, job);
  }
  const protectedJobs = ['3197751', '3197752', '3197753'];
  const queue = new Map([...original.values()].map(job => [job, protectedJobs.includes(job) ? 'RUNNING' : 'PENDING']));
  const calls = [], cancellations = [], updates = [];
  const mock = { execFileSync(command, args, options) {
    if (command === 'sha256sum') return cp.execFileSync(command, args, options);
    if (command === 'scontrol') {
      if (args[0] === 'show') return 'ClusterName = fir\n';
      updates.push(args); return '';
    }
    if (command === 'sinfo') return 'gpu:h100:4';
    if (command === 'squeue') return [...queue].map(row => row.join('|')).join('\n');
    if (command === 'scancel') {
      assert.equal(args[0], '--state=PENDING');
      for (const job of args.slice(1)) {
        assert(!protectedJobs.includes(job));
        assert.equal(queue.get(job), 'PENDING');
        queue.delete(job); cancellations.push(job);
      }
      return '';
    }
    assert.equal(command, 'sbatch');
    const job = String(900000 + calls.length);
    calls.push({ job, args }); queue.set(job, 'PENDING'); return job;
  }};
  const context = vm.createContext({ require: n => n === 'child_process' ? mock : require(n),
    module: { exports: {} }, process: { argv: [] }, console: { log() {} } });
  vm.runInContext(fs.readFileSync(path.join(__dirname, 'pair_fir.js'), 'utf8'), context);
  context.main(temporary, 'cancel');
  assert.equal(cancellations.length, 335);
  assert.equal(queue.size, 3);
  context.main(temporary, 'submit');
  assert.equal(calls.length, 168);
  const schedule = JSON.parse(fs.readFileSync(path.join(batch, 'fir_pairs.json')));
  assert.equal(schedule.groups.reduce((n, g) => n + g.tasks.length, 0), 335);
  assert.deepEqual(schedule.external.map(row => row[1]).sort(), protectedJobs);
  const locations = new Map(schedule.external), finished = new Set();
  calls.forEach(({ job, args }, i) => {
    const group = schedule.groups[i];
    for (const flag of ['--account=def-bghaddar', '--mem=11G', '--cpus-per-task=1', '--gpus-per-node=h100:1']) assert(args.includes(flag));
    assert(!args.some(a => a.startsWith('--array')));
    const script = args.findIndex(a => a.endsWith('/paired_job.sh'));
    assert.deepEqual(args.slice(script + 4), group.tasks);
    assert(args.includes('--time=02:00:00'));
    const required = [...new Set(group.tasks.flatMap(id => plan.tasks.find(t => t.id === id).requires))];
    const deps = [...new Set(required.map(id => locations.get(id)))];
    assert.equal(args.find(a => a.startsWith('--dependency=')), deps.length ? '--dependency=afterok:' + deps.join(':') : undefined);
    for (const id of group.tasks) locations.set(id, job);
  });
  context.main(temporary, 'submit');
  assert.equal(calls.length, 168, 'Repeated run duplicated active paired jobs');
  const first = schedule.groups[0].tasks[0], task = plan.tasks.find(t => t.id === first);
  const marker = path.join(temporary, plan.results, 'jobs', first);
  fs.mkdirSync(marker, { recursive: true });
  fs.writeFileSync(path.join(marker, '.done'), JSON.stringify({ protocol_hash: schedule.protocol_hash, task }));
  queue.delete(calls[0].job);
  context.main(temporary, 'submit');
  assert.equal(calls.length, 169, 'Stopped pair did not resume');
  assert.deepEqual(calls.at(-1).args.slice(-1), schedule.groups[0].tasks.slice(1));
  assert(calls.at(-1).args.includes('--time=02:00:00'));
  assert.equal(updates.length, 9, 'Paired crafting dependencies were not repaired');
  for (const job of protectedJobs) assert.equal(queue.get(job), 'RUNNING');
  console.log('PASS: 335 pending cancellations; three running jobs untouched; 168 paired Ghaddar/H100/11G jobs; dependencies, repeat skip and partial resume.');
} finally {
  fs.rmSync(temporary, { recursive: true, force: true });
}
