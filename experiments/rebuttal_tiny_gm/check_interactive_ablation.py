"""Small fixture: distinct methods, saved-poison reuse, and interrupted-run resume."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from interactive_ablation import VARIANTS, configuration, run, run_victim


with TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
    root = Path(temp)
    batch = root / 'batch'
    batch.mkdir()
    cell = dict(id='alligator_tarantula_b0005_basis', component='BASIS',
                targets=[8820, 3271, 6824, 7813, 4531],
                args=dict(budget=0.0005, lambda_margin=1, num_victims=5, victim_epochs=120))
    plan = dict(cells=[cell], results='results', data_bytes=1)
    (batch / 'plan.json').write_text(json.dumps(plan))
    (batch / 'code.sha256').write_text('fixture')
    source_hash = hashlib.sha256((batch / 'plan.json').read_bytes() +
                                 (batch / 'code.sha256').read_bytes()).hexdigest()

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    save(root / 'results/preflight.json', dict(complete=True, protocol_hash=source_hash))
    (root / 'data').mkdir()
    (root / 'data/tinyimagenet.pt').write_bytes(b'x')
    for index in range(20):
        model = root / 'results/surrogates' / f'net_{index}.pt'
        save(model.with_suffix('.json'), dict(epochs=150, identity=dict(
            protocol_hash=source_hash, target='surrogate', victim=index)))
        model.write_bytes(b'x')

    configs = [configuration(plan, source_hash, v) for v in VARIANTS]
    assert len({c['id'] for c, h in configs}) == len({h for c, h in configs}) == len(VARIANTS)
    assert configs[-1] == (cell, source_hash) and plan['cells'][0] == cell
    assert [(c['component'], c['args']['lambda_margin']) for c, h in configs] == list(VARIANTS.values())
    assert configuration(plan, source_hash, 'basis_lam10')[0]['args']['lambda_margin'] == 10
    crafted, pools = [], []

    def pool(*args, count):
        pools.append(count)
        return [object()] * count

    def poison(cell, ctx, nets, target, directory, identity):
        path = directory / 'poison.pt'
        if path.exists():
            assert json.loads(path.read_text()) == identity
        else:
            assert len(nets) == 20
            save(path, identity)
            crafted.append((cell['id'], target))
            if len(crafted) == 1:
                raise InterruptedError('fixture interruption')

    replay = SimpleNamespace(save=save, context=lambda *a: (None, {}), pool=pool,
        poison=poison, check_time=lambda: None,
        target_directory=lambda r, c, t: r / 'cells' / c['id'] / f'target_{t}')
    worker = SimpleNamespace(replay=replay)
    for variant in VARIANTS:
        if variant == 'basis_lam1':
            for target in cell['targets']:
                save(replay.target_directory(root / 'results', cell, target) / 'poison.pt',
                     dict(protocol_hash=source_hash, cell=cell['id'], target=target))
        try:
            run(worker, root, batch, variant)
        except InterruptedError:
            run(worker, root, batch, variant)
        before = len(crafted), len(pools)
        run(worker, root, batch, variant)
        assert before == (len(crafted), len(pools))
    assert len(crafted) == 5 * (len(VARIANTS) - 1) and pools == [20] * len(VARIANTS)
    trained, aggregated = [], []

    def train_victim(task, selected_cell, ctx, results, protocol_hash):
        target, victim = task['targets'][0], task['victim']
        directory = replay.target_directory(results, selected_cell, target)
        identity = dict(protocol_hash=protocol_hash, cell=selected_cell['id'], target=target)
        assert json.loads((directory / 'poison.pt').read_text()) == identity
        result = directory / f'victim_{victim}.json'
        if not result.exists():
            save(result, dict(identity=dict(identity, victim=victim)))
            trained.append((selected_cell['id'], target, victim))

    worker.train_victim = train_victim
    replay.aggregate = lambda p, r, h: aggregated.append((p['cells'][0]['id'], h))
    for variant in ('M', 'R', 'basis_lam0p1', 'basis_lam10'):
        before = len(crafted), len(pools)
        run_victim(worker, root, batch, variant, 8820, 0)
        run_victim(worker, root, batch, variant, 8820, 0)
        assert before == (len(crafted), len(pools))
    assert len(trained) == 4 and len(aggregated) == 8
    corrupt = root / 'results/cells/alligator_tarantula_b0005_M/craft_interactive.json'
    corrupt.write_text('{}')
    try:
        run(worker, root, batch, 'M')
    except AssertionError:
        pass
    else:
        raise AssertionError('An incompatible cache was accepted')

print('PASS: distinct configurations, saved-poison victim training, resume, lambda=1 reuse, and cache identity.')
