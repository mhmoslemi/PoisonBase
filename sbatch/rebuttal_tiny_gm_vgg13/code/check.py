"""Small GPU checks inside surrogate job zero, before downstream crafting."""
import gc
import json
import resource

import torch
import final_update as fu
import replay
from victim_aug import VictimAugmenter


def run(plan, ctx, results, protocol_hash):
    marker = results / 'preflight.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert saved['protocol_hash'] == protocol_hash and saved['complete']
        return
    assert len(ctx['train_labs']) == 100000 and len(ctx['test_labs']) == 10000
    for cell in plan['cells']:
        own, true = fu.parse_pair(cell['args']['class_pair'], ctx['class_names'])
        assert int((ctx['train_labs'] == own).sum()) == 500
        assert len(cell['targets']) == 6 and len(set(cell['targets'])) == 6
        assert all(int(ctx['test_labs'][t]) == true for t in cell['targets'])
    sample = plan['cells'][0]
    labels = torch.full((1000,), -1, dtype=torch.long, device=ctx['device'])
    labels[:500] = ctx['y_adv']
    fake = dict(ctx, train_labs=labels)
    select_function = fu.select_base_ours
    try:
        for cell in plan['cells']:
            cell = dict(cell, args=dict(cell['args'], budget=cell['args']['budget'] * 100))
            target = cell['targets'][0]
            count = round(cell['args']['budget'] * len(labels))

            def selected(*args, **kwargs):
                assert len(args[0]) == 20 and args[5] == count and args[6] == 1
                assert kwargs['distance_margin_coef'] is None and kwargs['base_dist'] == 'cosine'
                return torch.arange(count, device=ctx['device'])

            fu.select_base_ours = selected
            actual = replay.select(cell, fake, [None] * 20, target)
            if cell['component'] == 'random':
                generator = torch.Generator().manual_seed(42 * 100003 + target)
                expected = torch.randperm(500, generator=generator)[:count]
                assert torch.equal(actual.cpu(), expected)
            else:
                assert torch.equal(actual, torch.arange(count, device=ctx['device']))
    finally:
        fu.select_base_ours = select_function
    old = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    checkpoint = results / 'preflight_victim.pt'
    try:
        settings = dict(sample['args'], victim_epochs=2, victim_lr=0.01,
                        victim_bs=4, victim_decay=[1])
        net = fu.build_network('VGG13BN', 3, 200, (64, 64), ctx['device'], seed=123)
        assert net(ctx['train_imgs'][:4]).shape == (4, 200)
        initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        rng = replay.rng_state()

        def make():
            model = fu.build_network('VGG13BN', 3, 200, (64, 64), ctx['device'], seed=123)
            model.load_state_dict(initial)
            replay.restore_rng(rng)
            return model

        del net
        augmenter = VictimAugmenter('standard', ctx)
        images, labs = ctx['train_imgs'][:8], ctx['train_labs'][:8]
        reference = make()
        fu.train_from_scratch(reference, images, labs, 2, 0.01, 4, [1],
                              ctx['device'], augmenter=augmenter)
        expected = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
        del reference
        first = make()
        identity = dict(test='augmentation-SGD-resume', target=0, victim=0)
        replay.fit(first, images, labs, dict(settings, victim_epochs=1), checkpoint,
                   identity, augmenter=augmenter)
        del first
        resumed = make()
        replay.fit(resumed, images, labs, settings, checkpoint, identity, augmenter=augmenter)
        for key, value in resumed.state_dict().items():
            assert torch.allclose(value.cpu(), expected[key], atol=3e-6, rtol=2e-5), key
        del resumed, expected, initial, rng
        checkpoint.unlink()
    finally:
        torch.backends.cudnn.deterministic = old
        gc.collect()
        torch.cuda.empty_cache()
    replay.save(marker, dict(protocol_hash=protocol_hash, complete=True,
        checks=['target labels', 'Rand draws', 'BASIS lambda=1 dispatch',
                '64x64 VGG13BN head', 'augmented SGD checkpoint resume'],
        peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
    print('PREFLIGHT PASSED', flush=True)
