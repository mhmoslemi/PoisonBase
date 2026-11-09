const fs = require('fs'), path = require('path'), os = require('os');
const assert = require('assert'), cp = require('child_process'), vm = require('vm'), crypto = require('crypto');
const root = path.resolve(__dirname, '../..');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'tiny-fir-finish-'));
const project = path.join(temporary, 'project'), relative = 'sbatch/rebuttal_tiny_gm_vgg13_5x5';
const batch = path.join(project, relative);
fs.mkdirSync(batch, { recursive: true });
try {
  for (const name of ['plan.json', 'code.sha256', 'job.sh', 'code', 'submissions_fir.tsv'])
    fs.cpSync(path.join(root, relative, name), path.join(batch, name), { recursive: true });
  fs.mkdirSync(path.join(project, 'data'));
  fs.symlinkSync(path.join(root, 'data/tinyimagenet.pt'), path.join(project, 'data/tinyimagenet.pt'));
  const bytes = fs.readFileSync(path.join(batch, 'plan.json')), plan = JSON.parse(bytes);
  const hash = crypto.createHash('sha256').update(bytes).update(fs.readFileSync(path.join(batch, 'code.sha256'))).digest('hex');
  const source = new Map(fs.readFileSync(path.join(batch, 'submissions_fir.tsv'), 'utf8').trim().split('\n').map(l => l.split('\t').slice(0, 2)));
  for (let i = 15; i <= 18; i++) {
    const id = 'surrogate_' + i, job = String(3199654 + i - 15);
    fs.appendFileSync(path.join(batch, 'submissions_fir.tsv'), [id, job, '-', 'test'].join('\t') + '\n');
    source.set(id, job);
  }
  const done = task => {
    const directory = path.join(project, plan.results, 'jobs', task.id);
    fs.mkdirSync(directory, { recursive: true });
    fs.writeFileSync(path.join(directory, '.done'), JSON.stringify({ protocol_hash: hash, task }));
    if (task.stage === 'surrogate') {
      const model = path.join(project, plan.results, 'surrogates', 'net_' + task.surrogates[0]);
      fs.mkdirSync(path.dirname(model), { recursive: true });
      fs.writeFileSync(model + '.pt', 'test placeholder');
      fs.writeFileSync(model + '.json', JSON.stringify({ epochs: 150, seed: 42000 + task.surrogates[0], identity: { protocol_hash: hash, victim: task.surrogates[0] } }));
    }
  };
  for (const task of plan.tasks.slice(0, 3)) done(task);
  const queue = new Map(plan.tasks.filter(t => t.stage === 'surrogate').slice(3).map(t => [source.get(t.id), t.surrogates[0] < 12 ? 'RUNNING' : 'PENDING']));
  assert.equal([...queue.values()].filter(state => state === 'RUNNING').length, 9);
  assert.equal([...queue.values()].filter(state => state === 'PENDING').length, 8);
  const calls = [], updates = [];
  const mock = { execFileSync(command, args, options) {
    if (command === 'sha256sum') return cp.execFileSync(command, args, options);
    if (command === 'scontrol') {
      if (args[0] === 'show') return 'ClusterName = fir\n';
      updates.push(args); return '';
    }
    if (command === 'sinfo') return 'gpu:h100:4';
    if (command === 'squeue') return [...queue].map(row => row.join('|')).join('\n');
    assert.equal(command, 'sbatch', 'Only submission/read/dependency update commands are allowed');
    const job = String(910000 + calls.length);
    calls.push({ job, args }); queue.set(job, 'PENDING'); return job;
  }};
  const context = vm.createContext({ require: n => n === 'child_process' ? mock : require(n),
    module: { exports: {} }, process: { argv: [] }, console: { log() {} } });
  vm.runInContext(fs.readFileSync(path.join(__dirname, 'finish_fir.js'), 'utf8'), context);
  context.main(project);
  assert.equal(calls.length, 40);
  const runtime = path.join(temporary, 'tiny_gm_vgg13_fir_runtime');
  const schedule = JSON.parse(fs.readFileSync(path.join(runtime, 'finish_schedule.json')));
  assert.equal(schedule.groups.filter(g => g.stage === 'craft').length, 6);
  assert.equal(schedule.groups.filter(g => g.stage === 'victim').length, 34);
  const locations = new Map(plan.tasks.filter(t => t.stage === 'surrogate').slice(3).map(t => [t.id, source.get(t.id)]));
  let victimTrials = 0;
  calls.forEach(({ job, args }, index) => {
    const group = schedule.groups[index];
    for (const flag of ['--account=def-bghaddar', '--mem=12G', '--cpus-per-task=1', '--gpus-per-node=h100:1', '--time=03:15:00']) assert(args.includes(flag));
    assert(!args.some(a => a.startsWith('--array')));
    const script = args.findIndex(a => a.endsWith('/paired_job.sh'));
    assert.equal(args[script + 3], '11700');
    assert.deepEqual(args.slice(script + 4), group.tasks);
    assert(group.tasks.length <= (group.stage === 'craft' ? 3 : 9));
    assert(args.find(a => a.startsWith('--output=')).startsWith('--output=' + runtime + '/logs/'));
    const required = [...new Set(group.tasks.flatMap(id => plan.tasks.find(t => t.id === id).requires))];
    const deps = [...new Set(required.filter(id => locations.has(id)).map(id => locations.get(id)))];
    assert.equal(args.find(a => a.startsWith('--dependency=')), '--dependency=afterok:' + deps.join(':'));
    for (const id of group.tasks) locations.set(id, job);
    if (group.stage === 'victim') victimTrials += group.tasks.length;
  });
  assert.equal(victimTrials, 300);
  assert.equal(schedule.groups.at(-1).tasks.length, 3);
  context.main(project);
  assert.equal(calls.length, 40, 'Repeat submission duplicated active jobs');
  const victimGroup = schedule.groups[6];
  for (const id of victimGroup.tasks.slice(0, 2)) done(plan.tasks.find(t => t.id === id));
  queue.delete(calls[6].job);
  context.main(project);
  assert.equal(calls.length, 41);
  assert.deepEqual(calls.at(-1).args.slice(-7), victimGroup.tasks.slice(2));
  const firstCraft = schedule.groups[0];
  done(plan.tasks.find(t => t.id === firstCraft.tasks[0]));
  queue.delete(calls[0].job);
  context.main(project);
  assert.equal(calls.length, 42);
  assert.deepEqual(calls.at(-1).args.slice(-2), firstCraft.tasks.slice(1));
  assert(updates.length > 0, 'Victim dependencies were not repaired after crafting resumed');
  const lastSurrogate = plan.tasks.filter(t => t.stage === 'surrogate').at(-1);
  queue.delete(source.get(lastSurrogate.id));
  assert.throws(() => context.main(project), /unfinished and has no queued\/running job/);
  assert.equal(calls.length, 42, 'Missing surrogate caused unwanted submissions');
  console.log('PASS: 6 crafting + 34 nine-victim jobs, 03:15/12G/H100/Ghaddar, full surrogate dependency gate, logs outside Git, duplicate skip and partial resume.');
} finally {
  fs.rmSync(temporary, { recursive: true, force: true });
}
