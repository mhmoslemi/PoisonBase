"""Small fixture check: target isolation, cache reuse, and parent completion."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from craft_target import run_target


def check(component, expected_count):
    with TemporaryDirectory() as temp:
        root = Path(temp)
        batch = root / 'batch'
        batch.mkdir()
        task = dict(id='craft_example', stage='craft', cell='example', targets=[11, 22])
        cell = dict(id='example', component=component)
        plan = dict(results='results', tasks=[task], cells=[cell])
        (batch / 'plan.json').write_text(json.dumps(plan))
        (batch / 'code.sha256').write_text('fixture')
        protocol_hash = hashlib.sha256((batch / 'plan.json').read_bytes() +
                                      (batch / 'code.sha256').read_bytes()).hexdigest()
        results = root / 'results'
        results.mkdir()
        (results / 'preflight.json').write_text(json.dumps(
            dict(protocol_hash=protocol_hash, complete=True)))
        crafted, pool_counts = [], []

        def directory(cache, selected_cell, target):
            return cache / 'cells' / selected_cell['id'] / f'target_{target}'

        def save(path, value):
            path.write_text(json.dumps(value))

        def pool(*args, count):
            pool_counts.append(count)
            return [object()] * count

        def poison(selected_cell, ctx, nets, target, folder, identity, require_saved=False):
            path = folder / 'poison.pt'
            if path.exists():
                assert json.loads(path.read_text()) == identity
                return
            assert not require_saved and len(nets) == expected_count
            assert identity == dict(protocol_hash=protocol_hash, cell='example', target=target)
            folder.mkdir(parents=True, exist_ok=True)
            save(path, identity)
            crafted.append(target)

        replay = SimpleNamespace(save=save, context=lambda *a: (None, {}), pool=pool,
            poison=poison, target_directory=directory, check_time=lambda: None)
        worker = SimpleNamespace(replay=replay)
        run_target(worker, root, batch, task['id'], 11)
        assert crafted == [11] and pool_counts == [expected_count]
        parent = results / 'jobs' / task['id'] / '.done'
        assert not parent.exists()
        run_target(worker, root, batch, task['id'], 11)
        assert crafted == [11] and pool_counts == [expected_count]
        run_target(worker, root, batch, task['id'], 22)
        assert crafted == [11, 22] and pool_counts == [expected_count] * 2
        assert json.loads(parent.read_text()) == dict(protocol_hash=protocol_hash, task=task)
        try:
            run_target(worker, root, batch, task['id'], 33)
        except AssertionError:
            pass
        else:
            raise AssertionError('An unrelated target was accepted')
        child = results / 'jobs' / 'craft_example_target_11' / 'config.json'
        child.write_text('{}')
        try:
            run_target(worker, root, batch, task['id'], 11)
        except AssertionError:
            pass
        else:
            raise AssertionError('An incompatible saved config was accepted')


if __name__ == '__main__':
    check('random', 5)
    check('BASIS', 20)
    print('PASS: target isolation, saved-poison reuse, parent completion, identity checks.', flush=True)
