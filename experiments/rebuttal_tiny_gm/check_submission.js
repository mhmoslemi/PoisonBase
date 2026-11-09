// Check cluster flags, dependencies and repeat submissions without Slurm/GPU work.
const fs = require('fs'), path = require('path'), os = require('os');
const vm = require('vm'), assert = require('assert'), cp = require('child_process');
const root = path.resolve(__dirname, '../..');
const relative = 'sbatch/rebuttal_tiny_gm_vgg13_5x5';
const source = fs.readFileSync(path.join(__dirname, 'submit.js'), 'utf8');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'tiny-submit-check-'));
try {
  for (const site of ['vulcan', 'killarney', 'fir']) {
    const project = path.join(temporary, site), batch = path.join(project, relative);
    fs.mkdirSync(batch, { recursive: true });
    for (const name of ['plan.json', 'job.sh', 'code.sha256', 'code'])
      fs.cpSync(path.join(root, relative, name), path.join(batch, name), { recursive: true });
    const shared = path.join(project, 'shared');
    fs.mkdirSync(shared);
    fs.symlinkSync(path.join(root, 'data/tinyimagenet.pt'), path.join(shared, 'tinyimagenet.pt'));
    if (site === 'vulcan') {
      fs.mkdirSync(path.join(project, 'data'));
      fs.symlinkSync(path.join(shared, 'tinyimagenet.pt'), path.join(project, 'data/tinyimagenet.pt'));
    } else fs.writeFileSync(path.join(batch, 'submissions.tsv'), 'surrogate_00\t99999\t-\n');
    const calls = [];
    const updates = [];
    let queue = '';
    const mock = { execFileSync(command, args, options) {
      if (command === 'sha256sum') return cp.execFileSync(command, args, options);
      if (command === 'scontrol') {
        if (args[0] === 'show') return 'ClusterName = ' + site + '\n';
        assert.equal(args[0], 'update');
        updates.push(args);
        return '';
      }
      if (command === 'sinfo') return site === 'fir' ? 'gpu:h100:4' : 'gpu:l40s:4';
      if (command === 'sacctmgr') {
        assert.equal(site, 'vulcan');
        return 'aip-bghaddar|\naip-yiweilu|';
      }
      if (command === 'squeue') return queue;
      assert.equal(command, 'sbatch');
      const id = String(50000 + calls.length);
      calls.push({ id, args });
      return id;
    }};
    const context = vm.createContext({
      require: name => name === 'child_process' ? mock : require(name), module: { exports: {} },
      process: { env: { DATA_ROOT: shared } }, console: { log() {} }
    });
    vm.runInContext(source, context);
    context.main(project, site);
    const plan = JSON.parse(fs.readFileSync(path.join(batch, 'plan.json')));
    assert.equal(calls.length, 338);
    assert.equal(plan.tasks.filter(t => t.stage === 'victim').length, 300);
    for (const cell of plan.cells) {
      assert.equal(cell.targets.length, 5);
      assert.equal(cell.args.num_targets, 5);
      assert.equal(cell.args.num_victims, 5);
      assert.equal(cell.args.surrogate_epochs, 150);
      assert.equal(cell.args.victim_epochs, 120);
      const trials = plan.tasks.filter(t => t.stage === 'victim' && t.cell === cell.id);
      assert.equal(trials.length, 25);
      assert.equal(new Set(trials.map(t => t.targets[0] + '_' + t.victim)).size, 25);
    }
    const locations = new Map();
    let lu = 0;
    calls.forEach(({ id, args }, index) => {
      const task = plan.tasks[index];
      const account = site === 'fir' ? 'def-bghaddar' : site === 'killarney' ? 'aip-bghaddar' : task.account;
      assert(args.includes('--account=' + account));
      if (account === 'aip-yiweilu') lu++;
      const gpu = site === 'fir' ? '--gpus-per-node=h100:1' : site === 'killarney' ? '--gres=gpu:1' : '--gres=gpu:l40s:1';
      for (const flag of [gpu, '--cpus-per-task=1', '--mem=12G', '--time=' + task.time]) assert(args.includes(flag));
      assert(!args.some(a => a.startsWith('--array')), 'Array submission is forbidden');
      assert.equal(args.at(-1), task.id);
      const required = task.requires;
      const expected = required.length ? '--dependency=afterok:' + required.map(t => locations.get(t)).join(':') : undefined;
      assert.equal(args.find(a => a.startsWith('--dependency=')), expected);
      locations.set(task.id, id);
      queue += id + '|PENDING\n';
    });
    assert.equal(lu, site === 'vulcan' ? 6 : 0);
    context.main(project, site);
    assert.equal(calls.length, 338, 'Repeat submission duplicated jobs');
    queue = queue.replace(calls[0].id + '|PENDING\n', '');
    context.main(project, site);
    assert.equal(calls.length, 339, 'Single stopped job was not resumed independently');
    assert.equal(calls.at(-1).args.at(-1), plan.tasks[0].id);
    assert.equal(updates.length, 18, 'Craft dependencies were not repaired');
    for (const args of updates) assert(args.includes('Dependency=afterok:' + [calls.at(-1).id, ...calls.slice(1,20).map(c => c.id)].join(':')));
    console.log('PASS:', site, '338 individual jobs, 5x5 trials, resources/dependencies, repeat skip and single-job resume.');
  }
} finally {
  fs.rmSync(temporary, { recursive: true, force: true });
}
