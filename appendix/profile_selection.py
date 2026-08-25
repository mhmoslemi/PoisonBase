"""Time the selection stage on its own, for the two cost tables.

tab:selection-cost      surrogate training (one-time), candidate scoring, subset
                        selection, per-target total, peak memory
tab:selection-scaling   how selection time and memory grow with the surrogate
                        ensemble size K and the candidate-pool size

No poison optimization and no victim training happen here -- only the work the
selector itself does, so the numbers are comparable across selections.

    python appendix/profile_selection.py --model ResNet20BN --mode cost
    python appendix/profile_selection.py --model ResNet20BN --mode scaling
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import final_update as FU


def ns(a, **kw):
    d = dict(vars(a)); d.update(kw)
    return argparse.Namespace(**d)


def timed(fn, device, reps=3):
    """min over reps, with the cuda queue drained and peak memory reset."""
    torch.cuda.reset_peak_memory_stats(device) if device.startswith('cuda') else None
    best = float('inf')
    for _ in range(reps):
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
        t0 = time.time()
        out = fn()
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
        best = min(best, time.time() - t0)
    peak = (torch.cuda.max_memory_allocated(device) / 2**20) if device.startswith('cuda') else float('nan')
    return best, peak, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='CIFAR10')
    p.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    p.add_argument('--cache_dir', default='./cache')
    p.add_argument('--model', required=True)
    p.add_argument('--class_pair', default='dog-bird')
    p.add_argument('--pair_order', default='poison-target')
    p.add_argument('--budget', type=float, default=0.005)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_surrogates', type=int, default=20)
    p.add_argument('--surrogate_epochs', type=int, default=60)
    p.add_argument('--surrogate_lr', type=float, default=0.1)
    p.add_argument('--surrogate_bs', type=int, default=128)
    p.add_argument('--surrogate_decay', nargs='*', type=int, default=[35, 45])
    p.add_argument('--surrogate_wd', type=float, default=0.0)
    p.add_argument('--surrogate_aug', action='store_true', default=False)
    p.add_argument('--lambda_margin', type=float, default=1.0)
    p.add_argument('--base_dist', default='cosine')
    p.add_argument('--sel_alpha', type=float, default=2.0)
    p.add_argument('--sel_pool', type=float, default=3.0)
    p.add_argument('--sel_mu', type=float, default=1.0)
    p.add_argument('--dsa_strategy', default='color_crop_cutout_flip_scale_rotate')
    p.add_argument('--num_targets', type=int, default=10)
    p.add_argument('--mode', choices=['cost', 'scaling'], default='cost')
    p.add_argument('--out', default=None)
    a = p.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        sys.exit('profile_selection: no GPU visible -- run inside an allocation')

    ctx = FU.build_context(a, device)
    y_adv, _ = FU.parse_pair(a.class_pair, ctx['class_names'], a.pair_order)
    N_total = len(ctx['train_imgs'])
    N_p = int(round(a.budget * N_total))

    t0 = time.time()
    nets = FU.get_surrogates(a, ctx['train_imgs'], ctx['train_labs'], ctx['test_imgs'],
                             ctx['test_labs'], ctx['channel'], ctx['num_classes'],
                             ctx['im_size'], device, ctx['dsa_param'])
    surro_load = time.time() - t0

    # the one-time surrogate training cost, read from the cache the runs share
    d = FU.surrogate_dir(a)
    trained = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) if os.path.isdir(d) else 0

    pool_all = (ctx['train_labs'] == y_adv).nonzero(as_tuple=True)[0]
    targets = list(range(a.num_targets))
    rows = []

    if a.mode == 'cost':
        print('# %s  N_p=%d  pool=%d  K=%d  (surrogate load %.1f s, cache %.0f MB)'
              % (a.model, N_p, len(pool_all), a.num_surrogates, surro_load, trained / 2**20))
        print('# selection      score_s  subset_s  total_s  peak_MB')
        for name, mode in (('Random', None), ('Greedy', 'greedy'), ('DPP', 'dpp')):
            sc, ss, pk = [], [], []
            for t in targets:
                x_t = ctx['test_imgs'][t]
                gen = torch.Generator(device='cpu').manual_seed(a.seed * 100003 + t)
                if name == 'Random':
                    s, m, _ = timed(lambda: FU.select_base_random(ctx['train_labs'], y_adv, N_p, device, gen), device)
                    sc.append(0.0); ss.append(s); pk.append(m)
                    continue
                # scoring and subset selection share one call; time greedy (score +
                # top-N_p) and dpp (score + log-det greedy), the gap is the subset cost
                s, m, _ = timed(lambda: FU.select_base_ours(
                    nets, ctx['train_imgs'], ctx['train_labs'], x_t, y_adv, N_p,
                    a.lambda_margin, device, base_dist=a.base_dist), device)
                sc.append(s); pk.append(m)
                if mode == 'dpp':
                    s2, m2, _ = timed(lambda: FU.select_base_ours_div(
                        nets, ctx['train_imgs'], ctx['train_labs'], x_t, y_adv, N_p,
                        a.lambda_margin, device, base_dist=a.base_dist, mode='dpp',
                        pool=a.sel_pool, mu=a.sel_mu, alpha=a.sel_alpha), device)
                    ss.append(max(0.0, s2 - s)); pk[-1] = max(m, m2)
                else:
                    ss.append(0.0)
            mean = lambda v: sum(v) / len(v)
            rows.append(dict(selection=name, scoring=mean(sc), subset=mean(ss),
                             total=mean(sc) + mean(ss), peak_mb=max(pk)))
            print('  %-12s %8.3f %9.3f %8.3f %8.0f' % (name, mean(sc), mean(ss),
                                                       mean(sc) + mean(ss), max(pk)))
    else:
        print('# %s  N_p=%d  scaling of DPP selection' % (a.model, N_p))
        print('#   K   pool   time_s  peak_MB')
        for K in (1, 3, 5, 10, 20, 50):
            if K > len(nets):
                print('#   K=%d skipped: only %d surrogates cached' % (K, len(nets))); continue
            s, m, _ = timed(lambda: FU.select_base_ours_div(
                nets[:K], ctx['train_imgs'], ctx['train_labs'], ctx['test_imgs'][0], y_adv,
                N_p, a.lambda_margin, device, base_dist=a.base_dist, mode='dpp',
                pool=a.sel_pool, mu=a.sel_mu, alpha=a.sel_alpha), device)
            rows.append(dict(K=K, pool=len(pool_all), time=s, peak_mb=m))
            print('  %4d %6d %8.3f %8.0f' % (K, len(pool_all), s, m))
        for pool in (250, 500, 1000, 2500, 5000):
            keep = pool_all[:pool]
            if len(keep) < N_p:
                print('#   pool=%d skipped: smaller than N_p=%d' % (pool, N_p)); continue
            sub_imgs = ctx['train_imgs'].clone()
            mask = torch.ones(len(ctx['train_labs']), dtype=torch.bool, device=ctx['train_labs'].device)
            mask[pool_all] = False
            mask[keep] = True
            labs = ctx['train_labs'].clone()
            labs[~mask & (ctx['train_labs'] == y_adv)] = (y_adv + 1) % ctx['num_classes']
            s, m, _ = timed(lambda: FU.select_base_ours_div(
                nets, sub_imgs, labs, ctx['test_imgs'][0], y_adv, N_p, a.lambda_margin,
                device, base_dist=a.base_dist, mode='dpp', pool=a.sel_pool, mu=a.sel_mu,
                alpha=a.sel_alpha), device)
            rows.append(dict(K=len(nets), pool=pool, time=s, peak_mb=m))
            print('  %4d %6d %8.3f %8.0f' % (len(nets), pool, s, m))

    if a.out:
        with open(a.out, 'w') as f:
            json.dump({'model': a.model, 'mode': a.mode, 'N_p': N_p,
                       'pool': int(len(pool_all)), 'rows': rows}, f, indent=1)
        print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
