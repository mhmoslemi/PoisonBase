"""Small numerical checks; run only inside a Slurm allocation."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from scores import (hessian_vector_product, influence_input_scores,
                    lissa_update, loss_gradient, meta_weight_gradients,
                    normalize_meta_weights)


def check_numerics(device):
    torch.manual_seed(91)
    net = nn.Linear(3, 2, dtype=torch.float64).to(device).eval()
    images = torch.randn(5, 3, dtype=torch.float64, device=device)
    labels = torch.tensor([0, 1, 0, 1, 1], device=device)
    target = images[1:2] + 0.2
    target_label = torch.tensor([0], device=device)
    gt = loss_gradient(net, target, target_label)
    actual = meta_weight_gradients(net, images, labels, target, target_label)
    reference = []
    for image, label in zip(images, labels):
        gi = loss_gradient(net, image[None], label[None])
        reference.append(sum((a * b).sum() for a, b in zip(gi, gt)))
    torch.testing.assert_close(actual, torch.stack(reference), atol=1e-11, rtol=1e-10)
    values = torch.tensor([-2., 0., 1., 3.], device=device)
    torch.testing.assert_close(normalize_meta_weights(values),
                               torch.tensor([0., 0., .25, .75], device=device))
    assert not normalize_meta_weights(-values.abs()).any()

    parameters = tuple(net.parameters())
    sizes = [parameter.numel() for parameter in parameters]
    flat = torch.cat([parameter.detach().flatten() for parameter in parameters])

    def unpack(vector):
        result, start = [], 0
        for parameter, size in zip(parameters, sizes):
            result.append(vector[start:start + size].reshape_as(parameter))
            start += size
        return tuple(result)

    def objective(vector, x=images):
        weight, bias = unpack(vector)
        return F.cross_entropy(F.linear(x, weight, bias), labels)

    hessian = torch.autograd.functional.hessian(objective, flat)
    dense_gt = torch.cat([value.flatten() for value in gt])
    hvp = hessian_vector_product(net, images, labels, gt)
    torch.testing.assert_close(torch.cat([value.flatten() for value in hvp]),
                               hessian @ dense_gt, atol=1e-11, rtol=1e-10)
    scale, damping = 10., .1
    estimate = gt
    for _ in range(500):
        estimate = lissa_update(gt, estimate,
                                hessian_vector_product(net, images, labels, estimate),
                                scale, damping)
    solution = tuple(value / scale for value in estimate)
    exact = torch.linalg.solve(hessian + scale * damping * torch.eye(
        len(flat), dtype=flat.dtype, device=device), dense_gt)
    torch.testing.assert_close(torch.cat([value.flatten() for value in solution]),
                               exact, atol=1e-10, rtol=1e-9)
    score = influence_input_scores(net, images, labels, solution)
    reference = []
    for index in range(len(images)):
        image = images[index:index + 1].detach().requires_grad_(True)
        gradients = torch.autograd.grad(F.cross_entropy(net(image), labels[index:index + 1]),
                                        parameters, create_graph=True)
        interaction = sum((a * b).sum() for a, b in zip(gradients, unpack(exact)))
        expected = torch.autograd.grad(interaction, image)[0].abs().sum()
        reference.append(expected)
    torch.testing.assert_close(score, torch.stack(reference), atol=1e-11, rtol=1e-10)
    # Independent finite difference of the influence scalar w.r.t. one pixel.
    epsilon = 1e-5
    perturbed = images[0:1].clone()
    def interaction(x):
        gradient = loss_gradient(net, x, labels[0:1])
        return sum((a * b).sum().item() for a, b in zip(gradient, unpack(exact)))
    perturbed[0, 0] += epsilon
    plus = interaction(perturbed)
    perturbed[0, 0] -= 2 * epsilon
    minus = interaction(perturbed)
    image = images[0:1].detach().requires_grad_(True)
    gradients = torch.autograd.grad(F.cross_entropy(net(image), labels[0:1]),
                                    parameters, create_graph=True)
    scalar = sum((a * b).sum() for a, b in zip(gradients, unpack(exact)))
    derivative = torch.autograd.grad(scalar, image)[0][0, 0].item()
    assert abs((plus - minus) / (2 * epsilon) - derivative) < 1e-8
    print('PASS: explicit meta-weights, clipping/normalization, dense HVP, '
          'LiSSA versus exact inverse, input-influence scores and finite differences.', flush=True)


def benchmark(root, device):
    import final_update as fu
    from pathlib import Path
    for name in ('ConvNetBN', 'ResNet20BN'):
        net = fu.build_network(name, 3, 10, (32, 32), device).eval()
        path = Path(root) / 'cache/surrogates' / f'{name}_60ep_lr0.1_bs128_seed42/net_0.pt'
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        images = torch.randn(64, 3, 32, 32, device=device)
        labels = torch.randint(0, 10, (64,), device=device)
        gt = loss_gradient(net, images[:1], torch.tensor([5], device=device))
        for _ in range(3):
            hessian_vector_product(net, images[:32], labels[:32], gt)
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(30):
            hessian_vector_product(net, images[:32], labels[:32], gt)
        if device == 'cuda':
            torch.cuda.synchronize()
        per_hvp = (time.perf_counter() - started) / 30
        started = time.perf_counter()
        influence_input_scores(net, images, labels, gt)
        meta_weight_gradients(net, images, labels, images[:1], torch.tensor([5], device=device))
        if device == 'cuda':
            torch.cuda.synchronize()
        print(f'{name}: HVP(batch32)={per_hvp:.5f}s; '
              f'20 x 10000 recursion steps~{20*10000*per_hvp/3600:.2f}h; '
              f'score smoke={time.perf_counter()-started:.3f}s', flush=True)


def check_resume(device):
    import tempfile
    from pathlib import Path
    import worker
    torch.manual_seed(112)
    net = nn.Sequential(nn.Flatten(), nn.Linear(12, 6)).to(device).eval()
    images = torch.randn(12, 3, 2, 2, device=device)
    labels = torch.tensor([5] * 6 + [0, 1, 2, 3, 4, 5], device=device)
    ctx = dict(device=device, channel=3, train_imgs=images, train_labs=labels,
               test_imgs=images[:1] + .1, norm=lambda x: x)
    candidates = torch.arange(6, device=device)
    protocol = dict(seed=42, y_adv=5, surrogates=1,
                    meta=dict(step_size=1, candidate_batch=2),
                    influence=dict(scale=10, ridge=.1, max_scale=1280,
                                   depth=300, hessian_batch=4, candidate_batch=2))
    with tempfile.TemporaryDirectory(dir=__import__('os').environ['SLURM_TMPDIR']) as temp:
        for method in ('meta', 'influence'):
            task = dict(method=method, model='ConvNetBN')
            fresh, resumed = Path(temp) / 'fresh', Path(temp) / 'resumed'
            worker.DEADLINE = float('inf')
            worker.score_one(net, ctx, candidates, 0, task, 0, fresh, protocol, 'test')
            worker.DEADLINE = 0
            try:
                worker.score_one(net, ctx, candidates, 0, task, 0, resumed, protocol, 'test')
            except SystemExit as error:
                assert error.code == 75
            else:
                raise AssertionError('Expected a deliberate checkpoint pause')
            worker.DEADLINE = float('inf')
            worker.score_one(net, ctx, candidates, 0, task, 0, resumed, protocol, 'test')
            left = worker.load(worker.score_path(fresh, method, 'ConvNetBN', 0, 0))
            right = worker.load(worker.score_path(resumed, method, 'ConvNetBN', 0, 0))
            torch.testing.assert_close(left['values'], right['values'], atol=0, rtol=0)
            rank = worker.ranking(resumed, task, 0, protocol, 'test')
            assert set(rank.tolist()) == set(candidates.tolist())
            # Exercise the actual cached-selector -> prepare_poisons interface,
            # stubbing only crafting/training to keep this a tiny check.
            import argparse
            import csv
            old = {name: getattr(worker.fu, name) for name in (
                'main', 'prepare_poisons', 'select_base_criterion',
                'train_from_scratch', 'craft_selected_bases')}
            task.update(attack='fc', budget=.002, targets=[0])
            ctx['denorm'] = lambda x: (x + 2) / 4
            def craft(args, context, nets, indices, target, y_adv):
                assert torch.equal(indices.cpu(), rank[:2].cpu())
                base = context['denorm'](context['train_imgs'][indices]).clamp(0, 1)
                return base, (base + .001).clamp(0, 1), 0.
            def mini_main(args):
                directory = Path(args.out_dir) / worker.fu.build_run_name(args)
                directory.mkdir(parents=True, exist_ok=True)
                worker.fu.prepare_poisons(args, ctx, [], [], 0, 5, 2, str(directory), ({}, {}))
                with (directory / 'results.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=worker.fu.RESULT_FIELDS)
                    writer.writeheader()
                    for victim in range(6):
                        row = {name: 0 for name in worker.fu.RESULT_FIELDS}
                        row.update(model='ConvNetBN', attack='fc', budget=.002,
                                   target_idx=0, victim_id=victim, success=0,
                                   clean_test_acc=.8)
                        writer.writerow(row)
            try:
                worker.fu.main = mini_main
                worker.fu.craft_selected_bases = craft
                job_dir = Path(temp) / f'integration_{method}'
                worker.run_evaluation(argparse.Namespace(data='unused', cache='unused'),
                                      task, protocol, 'test', resumed, job_dir)
            finally:
                for name, function in old.items():
                    setattr(worker.fu, name, function)
    print('PASS: scorer checkpoint/resume is bitwise identical to uninterrupted scoring; '
          'ensemble ranking uses the complete cache.', flush=True)


def benchmark_real(root, device):
    import argparse
    import final_update as fu
    from pathlib import Path
    ctx = fu.build_context(argparse.Namespace(dataset='CIFAR10', data_path=str(Path(root) / 'data')), device)
    for name, target_id in [('ConvNetBN', 843), ('ResNet20BN', 5118)]:
        net = fu.build_network(name, 3, 10, (32, 32), device).eval()
        net.load_state_dict(torch.load(Path(root) / 'cache/surrogates' /
                            f'{name}_60ep_lr0.1_bs128_seed42/net_0.pt',
                            map_location=device, weights_only=True))
        rhs = loss_gradient(net, ctx['test_imgs'][target_id:target_id + 1],
                            torch.tensor([5], device=device))
        estimate = tuple(value.clone() for value in rhs)
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(200):
            indices = torch.randint(50000, (32,), device=device)
            estimate = lissa_update(rhs, estimate, hessian_vector_product(
                net, ctx['train_imgs'][indices], ctx['train_labs'][indices], estimate), 10., .01)
            if step in (49, 99, 199):
                norm = sum(value.double().square().sum() for value in estimate).sqrt().item()
                assert __import__('math').isfinite(norm), 'Non-finite LiSSA estimate on real inputs'
                print(f'{name}: real-data LiSSA step {step+1}/200 norm={norm:.5g}', flush=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        step_seconds = (time.perf_counter() - started) / 200
        print(f'{name}: measured full LiSSA step={step_seconds:.5f}s; '
              f'20-surrogate/10000-step target estimate={step_seconds*200000/60:.1f} min', flush=True)
        del net, estimate, rhs
        torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--resume-check', action='store_true')
    parser.add_argument('--real-timing', action='store_true')
    options = parser.parse_args()
    torch.set_num_threads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Numerical checks on', device, flush=True)
    if options.resume_check:
        check_resume(device)
    if options.real_timing:
        benchmark_real(options.root, device)
    if not options.resume_check and not options.real_timing:
        check_numerics(device)
    if options.root and not options.real_timing and not options.resume_check:
        benchmark(options.root, device)
