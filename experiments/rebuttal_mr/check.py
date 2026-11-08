"""Compute-node checks that gate all M/R experiments."""
import gc
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

import final_update as fu


def run(plan, options, results, mr):
    assert len(plan['cells']) == 12 and len(plan['tasks']) == 29
    assert sum(len(c['targets']) for c in plan['cells']) == 52
    assert sum(len(c['targets']) * c['args']['num_victims'] for c in plan['cells']) == 228
    checked = []
    work = Path(options.data).parent / 'preflight'
    work.mkdir(parents=True, exist_ok=True)
    for dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet']:
        cells = [c for c in plan['cells'] if c['args']['dataset'] == dataset]
        cell = next(c for c in cells if c['args']['attack'] == 'gradmatch')
        began = time.monotonic()
        _, ctx = mr.context(cell, options.data)
        for other in cells:
            _, target_class = fu.parse_pair(other['args']['class_pair'], ctx['class_names'])
            assert all(int(ctx['test_labs'][t]) == target_class for t in other['targets'])
        nets = mr.pool(cell, ctx, options.cache)
        target = cell['targets'][0]
        count = round(len(ctx['train_labs']) * cell['args']['budget'])
        for component in ['R', 'M']:
            variant = dict(cell, component=component)
            indices = mr.select(variant, ctx, nets, target)
            assert len(indices) == count
            # Compare the selected cost against an independent per-surrogate formula.
            with torch.no_grad():
                cls, actual, _ = fu._ours_pointwise_score(nets[:1], ctx['train_imgs'],
                    ctx['train_labs'], ctx['test_imgs'][target], ctx['y_adv'], 1,
                    ctx['device'], base_dist='cosine',
                    distance_margin_coef=1.0 if component == 'R' else 0.0)
                raw = []
                net = nets[0]
                emb = fu.embed_of(net)
                ft = emb(ctx['test_imgs'][target].unsqueeze(0))
                for ids in cls.split(512):
                    images = ctx['train_imgs'][ids]
                    if component == 'R':
                        raw.append(1 - F.cosine_similarity(emb(images), ft.expand(len(ids), -1), dim=1))
                    else:
                        logits = net(images)
                        own = logits[:, ctx['y_adv']].clone()
                        logits[:, ctx['y_adv']] = -float('inf')
                        raw.append(own - logits.max(1).values)
                assert torch.allclose(actual, fu.standardize(torch.cat(raw)), atol=2e-6, rtol=2e-5)
            print(f'CHECK {dataset} {component}: {len(indices)} unique class-correct bases', flush=True)

        args = cell['args']
        identity = dict(target=target, victim=0, test='SGD resume')
        mini = ctx['train_imgs'][:250]
        labels = ctx['train_labs'][:250]
        settings = dict(args, victim_epochs=2, victim_decay=[1])
        initial = fu.build_network(args['model'], ctx['channel'], ctx['num_classes'],
                                   ctx['im_size'], ctx['device'], seed=12345)
        initial_state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
        initial_rng = mr.rng_state()
        del initial

        def make():
            # The legacy ResNet factory reseeds from the clock; compare identical starts.
            net = fu.build_network(args['model'], ctx['channel'], ctx['num_classes'],
                                   ctx['im_size'], ctx['device'], seed=12345)
            net.load_state_dict(initial_state)
            mr.restore_rng(initial_rng)
            return net
        old_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True
        reference = make()
        fu.train_from_scratch(reference, mini, labels, 2, args['victim_lr'], 125, [1],
                              ctx['device'], weight_decay=0, aug=False)
        expected = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
        del reference
        checkpoint = work / f'{dataset}_victim.pt'
        first = make()
        mr.fit(first, mini, labels, dict(settings, victim_epochs=1), checkpoint, identity)
        del first
        resumed = make()
        mr.fit(resumed, mini, labels, settings, checkpoint, identity)
        for key, value in resumed.state_dict().items():
            assert torch.allclose(value.cpu(), expected[key], atol=3e-6, rtol=2e-5), key
        del resumed, expected, initial_state, initial_rng
        checkpoint.unlink()
        print(f'CHECK {dataset}: checkpointed SGD equals original uninterrupted SGD', flush=True)

        indices = mr.select(cell, ctx, nets, target)
        base01 = ctx['denorm'](ctx['train_imgs'][indices]).clamp(0, 1).detach()
        craft_nets = nets[:5]
        # grid_sample's CUDA backward is nondeterministic; test exact resume without DSA,
        # then separately exercise the source run's full-size augmented craft graph.
        kwargs = dict(dsa_strategy=None, dsa_param=ctx['dsa_param'],
                      lowmem=args['craft_lowmem'], chunk=args['craft_batch'])
        values = (craft_nets, base01, ctx['test_imgs'][target], ctx['y_adv'], ctx['norm'],
                  args['epsilon'], args['craft_alpha'], 2, 1, ctx['device'])
        fu.set_seed(76543)
        full, _ = fu.craft_gradmatch(*values, **kwargs)
        expected_delta = (full - base01).cpu()
        del full
        craft_checkpoint = work / f'{dataset}_craft.pt'

        class Interrupted(Exception):
            pass

        def interrupt(state):
            mr.save(craft_checkpoint, dict(state=state, numpy=np.random.get_state(),
                                          python=random.getstate()), tensor=True)
            raise Interrupted()

        fu.set_seed(76543)
        try:
            fu.craft_gradmatch(*values, **kwargs, checkpoint_callback=interrupt)
        except Interrupted:
            pass
        saved = mr.load(craft_checkpoint)
        np.random.set_state(saved['numpy'])
        random.setstate(saved['python'])
        resumed, _ = fu.craft_gradmatch(*values, **kwargs, resume_state=saved['state'])
        assert torch.allclose((resumed-base01).cpu(), expected_delta, atol=1e-6, rtol=0)
        del resumed, saved, expected_delta
        craft_checkpoint.unlink()
        adv, _ = fu.craft_gradmatch(*values, **dict(kwargs, dsa_strategy=args['dsa_strategy']))
        assert float((adv-base01).abs().max()) <= args['epsilon'] + 1e-6
        del adv
        if dataset == 'TinyImageNet':
            # Exercise full-size BP graphs too; no preflight poisons enter the results.
            adv, _ = fu.craft_fc(craft_nets, base01, ctx['test_imgs'][target], ctx['norm'],
                args['epsilon'], 2, args['craft_alpha'], ctx['device'], restarts=1, mode='sample')
            assert float((adv-base01).abs().max()) <= args['epsilon'] + 1e-6
            del adv
        torch.backends.cudnn.deterministic = old_deterministic
        record = dict(dataset=dataset, selection_surrogates=len(nets), crafting_surrogates=5,
                      poisons=count, seconds=time.monotonic()-began,
                      peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                      peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3)
        checked.append(record)
        print('PREFLIGHT PASS', json.dumps(record), flush=True)
        del ctx, nets, craft_nets, mini, labels, base01, values, raw, images, emb, net, ft, indices
        gc.collect()
        torch.cuda.empty_cache()
        mr.check_time()
    mr.save(results / 'preflight_checks.json', checked)
