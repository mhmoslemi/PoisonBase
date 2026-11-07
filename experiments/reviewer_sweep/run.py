"""Allocated-job entry point for the four reviewer experiment groups."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(os.environ.get('REVIEW_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.environ.get('REVIEW_CODE', str(ROOT)))
import numpy as np
import torch
import torch.nn.functional as F
import final_update as fu
from protocol import (MODELS, REPEATS, VERSION, cells, attack_cell, attack_args,
                      targets_for, subsets, atomic_json, fingerprint)
from defenses import (save_torch, losses, train_asd, train_trajectory,
                      detect_telltale, augment)

OUT = ROOT/'reviewer_20261106_v2_result'


def context(model):
    args = attack_args(model, os.environ.get('REVIEW_DATA', ROOT/'data'))
    ctx = fu.build_context(args, torch.device('cuda:0'))
    targets = targets_for(model)
    assert len(ctx['train_labs']) == 50000
    assert all(int(ctx['test_labs'][t]) == 2 for t in targets)
    labels = ctx['test_labs'].cpu().numpy()
    gen = np.random.default_rng(71307)
    val = np.concatenate([gen.permutation(np.setdiff1d(np.flatnonzero(labels==c), targets))[:100]
                          for c in range(10)])
    ctx['val_idx'] = torch.tensor(val, device=ctx['device'])
    ctx['eval_idx'] = torch.arange(len(labels), device=ctx['device'])
    ctx['targets'] = targets
    return args, ctx


def build(model, ctx, seed):
    return fu.build_network(model,3,10,(32,32),ctx['device'],seed=seed)


def pool_path(model, fraction, repeat, ctx):
    if fraction == 1:
        return ROOT/'cache'/'surrogates'/f'{model}_60ep_lr0.1_bs128_seed42'
    _, visible, _ = subsets(ctx['train_labs'].cpu().numpy(), repeat)
    digest = fingerprint(visible[fraction].tolist())
    return OUT/'surrogates'/f'{model}_f{fraction}_r{repeat}_{digest}'


def load_pool(model, fraction, repeat, ctx):
    directory = pool_path(model,fraction,repeat,ctx)
    nets=[]
    for i in range(20):
        net=build(model,ctx,1042+i)
        net.load_state_dict(torch.load(directory/f'net_{i}.pt',map_location=ctx['device'],weights_only=True))
        nets.append(net.eval())
    return nets


def pretrain(model, fraction, repeat, part):
    args,ctx=context(model)
    _,visible,_=subsets(ctx['train_labs'].cpu().numpy(),repeat)
    ids=torch.tensor(visible[fraction],device=ctx['device'])
    directory=pool_path(model,fraction,repeat,ctx)
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/'protocol.json',dict(version=VERSION,model=model,fraction=fraction,
                repeat=repeat,indices=ids.cpu().tolist(),epochs=60,decay=[35,45],seed_base=1042))
    for i in range(part*5,(part+1)*5):
        path=directory/f'net_{i}.pt'
        if path.exists():
            continue
        begin=time.monotonic()
        net=build(model,ctx,1042+i)
        fu.train_from_scratch(net,ctx['train_imgs'][ids],ctx['train_labs'][ids],60,.1,128,[35,45],ctx['device'])
        save_torch(path,net.state_dict())
        atomic_json(directory/f'net_{i}.json',dict(seconds=time.monotonic()-begin,
                    max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    clean_accuracy=accuracy(net,ctx)))
        del net


def accuracy(net,ctx):
    ids=ctx['eval_idx']
    return fu.test_acc(net,ctx['test_imgs'][ids],ctx['test_labs'][ids],bs=256)


def metrics(net,ctx):
    net.eval()
    with torch.no_grad():
        pred=net(ctx['test_imgs'][list(ctx['targets'])]).argmax(1).cpu().tolist()
    return dict(clean_accuracy=accuracy(net,ctx),
                target_predictions=dict(zip(map(str,ctx['targets']),pred)))


def ordinary(net,ctx):
    return fu.train_from_scratch(net,ctx['train_imgs'],ctx['train_labs'],60,.1,125,[40],ctx['device'])


def asd_supervised(net,ctx,directory):
    """ASD schedule/optimizer/augmentation control, with all labels retained."""
    from defenses import PoolBatches, rng_state, restore_rng
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/'checkpoint.pt'
    x,y=ctx['train_imgs'],ctx['train_labs']
    opt=torch.optim.Adam(net.parameters(),lr=.002)
    start=0
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location=ctx['device'],weights_only=False)
        net.load_state_dict(state['net']); opt.load_state_dict(state['opt'])
        start=state['epoch']; restore_rng(state['rng'])
    for ep in range(start,120):
        net.train(); batches=PoolBatches(torch.arange(len(y),device=y.device),64)
        for _ in range(1024):
            idx=batches.next(); opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(augment(x[idx])),y[idx]).backward(); opt.step()
        if (ep+1)%10==0:
            save_torch(checkpoint,dict(net=net.state_dict(),opt=opt.state_dict(),epoch=ep+1,rng=rng_state()))
            print('ASD supervised control',ep+1,flush=True)
    return net.eval()


def control(model,repeat,kind):
    args,ctx=context(model)
    directory=OUT/'controls'/model/f'{kind}_r{repeat}'
    if (directory/'result.json').exists():
        return
    net=build(model,ctx,4200000+repeat)
    start=time.monotonic()
    extra={}
    trusted,_,_=subsets(ctx['train_labs'].cpu().numpy(),0)
    if kind=='plain':
        ordinary(net,ctx)
    elif kind=='asd':
        train_asd(net,ctx['train_imgs'],ctx['train_labs'],
                  torch.tensor(trusted,device=ctx['device']),directory)
    elif kind=='asd_supervised':
        asd_supervised(net,ctx,directory)
    else:
        val=ctx['val_idx']
        _,traces,validation=train_trajectory(net,ctx['train_imgs'],ctx['train_labs'],
            ctx['test_imgs'][val],ctx['test_labs'][val],directory)
        suspicious,info=detect_telltale(traces,validation,ctx['device'],4200000+repeat,directory)
        save_torch(directory/'clean_trajectories.pt',torch.from_numpy(traces))
        atomic_json(directory/'detection.json',dict(info,clean_false_positive_rate=float(suspicious.mean())))
        extra['before_filter']=metrics(net,ctx)
        kept=torch.tensor(np.flatnonzero(~suspicious),device=ctx['device'])
        del net
        net=build(model,ctx,4200000+repeat)
        fu.train_from_scratch(net,ctx['train_imgs'][kept],ctx['train_labs'][kept],
                             60,.1,125,[40],ctx['device'])
    result=metrics(net,ctx)
    result.update(extra)
    result.update(seconds=time.monotonic()-start,version=VERSION,model=model,kind=kind,repeat=repeat)
    atomic_json(directory/'result.json',result)


def get_poison(cell,repeat,tidx,args,ctx,nets):
    # One fixed attacker-visible subset and poison set per table cell, shared
    # across the five independently seeded victims.
    poison_repeat=0
    directory=OUT/'cells'/f"{cell['id']:03d}"/'poisons'/f'r{poison_repeat}'
    path=directory/f'target_{tidx}.pt'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/f'target_{tidx}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            result=torch.load(path,map_location='cpu',weights_only=False)
            assert result['version']==VERSION and result['cell']==cell
            return result
        begin=time.monotonic()
        trusted,visible,candidates=subsets(ctx['train_labs'].cpu().numpy(),poison_repeat)
        candidate=torch.tensor(candidates[cell['pool']],device=ctx['device'])
        n=round(50000*cell['budget'])
        assert len(candidate)>=n
        if cell['fraction']==cell['pool'] and cell['fraction']!=1:
            assert set(candidate.cpu().tolist()) <= set(visible[cell['fraction']].tolist())
        selection_seed=42*100003+tidx+poison_repeat*10000019
        fu.set_seed(selection_seed)
        if cell['selector']=='random':
            idx=candidate[torch.randperm(len(candidate),device=ctx['device'])[:n]]
        elif cell['selector'].startswith('wo_'):
            scores=torch.load(OUT/'wicked'/f"{cell['selector']}.pt",weights_only=False)
            values=torch.full((50000,),-torch.inf,device=ctx['device'])
            values[scores['indices'].to(ctx['device'])]=scores['scores'].to(ctx['device'])
            idx=candidate[torch.argsort(values[candidate],descending=True,stable=True)[:n]]
        else:
            labels=torch.full_like(ctx['train_labs'],-1); labels[candidate]=5
            coefficient={'margin':0.,'similarity':1.,'basis':.5}[cell['selector']]
            idx=fu.select_base_ours(nets,ctx['train_imgs'],labels,ctx['test_imgs'][tidx],
                                   5,n,1.,ctx['device'],base_dist='cosine',distance_margin_coef=coefficient)
        assert len(idx.unique())==n and not np.intersect1d(idx.cpu().numpy(),trusted).size
        selection_seconds=time.monotonic()-begin
        # Identical final-crafting RNG for every selector with this target/draw.
        fu.set_seed(selection_seed)
        begin=time.monotonic()
        base,adv,obj=fu.craft_selected_bases(args,ctx,nets[:5],idx,ctx['test_imgs'][tidx],5)
        delta=(adv-base).detach().cpu()
        assert torch.isfinite(delta).all() and float(delta.abs().max()) <= 8/255+1e-6
        result=dict(version=VERSION,cell=cell,target=tidx,repeat=poison_repeat,
                    indices=idx.cpu(),delta=delta,objective=float(obj),
                    selection_seconds=selection_seconds,craft_seconds=time.monotonic()-begin,
                    candidate_hash=fingerprint(candidate.cpu().tolist()),
                    surrogate_subset_hash=fingerprint(visible[cell['fraction']].tolist()))
        save_torch(path,result)
        print('Crafted',cell['id'],tidx,poison_repeat,result['craft_seconds'],flush=True)
        return result


def detection_metrics(suspect,poison,labels):
    suspect=np.asarray(suspect,dtype=bool); poison=np.asarray(poison,dtype=bool)
    return dict(poison_recall=float(suspect[poison].mean()) if poison.any() else None,
                clean_false_positive_rate=float(suspect[~poison].mean()),
                dog_clean_false_positive_rate=float(suspect[(~poison)&(labels==5)].mean()))


def run_cell(cell_id,part):
    cell=cells()[cell_id]; base=attack_cell(cell)
    args,ctx=context(cell['model'])
    args.budget=cell['budget']
    all_targets=targets_for(cell['model'])
    targets=all_targets[part*2:(part+1)*2] if cell['defense']=='none' else all_targets[part:part+1]
    assert targets
    folder=OUT/'cells'/f'{cell_id:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'protocol.json',dict(version=VERSION,cell=cell,targets=all_targets,
                repeats=REPEATS,victim_epochs=60,victim_batch_size=125,
                victim_decay=[40],craft_steps=250,craft_ensemble=5,
                selection_ensemble=20,epsilon=8/255,validation_size=1000,cta_size=10000))
    with (folder/f'part_{part}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        nets=None
        for repeat in REPEATS:
            pending=[t for t in targets if not (folder/f't{t}_r{repeat}'/'result.json').exists()]
            if not pending:
                continue
            if cell['defense']=='none':
                if nets is None:
                    del nets
                    nets=load_pool(cell['model'],cell['fraction'],0,ctx)
            trusted,_,_=subsets(ctx['train_labs'].cpu().numpy(),0)
            for tidx in pending:
                directory=folder/f't{tidx}_r{repeat}'; directory.mkdir(parents=True,exist_ok=True)
                if cell['defense']=='none':
                    poison=get_poison(base,repeat,tidx,args,ctx,nets)
                else:
                    poison=torch.load(OUT/'cells'/f"{base['id']:03d}"/'poisons'/'r0'/f'target_{tidx}.pt',
                                      weights_only=False)
                    assert poison['version']==VERSION and poison['cell']==base
                idx=poison['indices'].to(ctx['device'])
                clean=ctx['train_imgs'][idx].clone()
                original01=ctx['denorm'](clean).clamp(0,1)
                ctx['train_imgs'][idx]=ctx['norm']((original01+poison['delta'].to(ctx['device'])).clamp(0,1))
                pmask=np.zeros(50000,dtype=bool); pmask[poison['indices'].numpy()]=True
                begin=time.monotonic(); info={}
                try:
                    victim_seed=42*100000+tidx*100+repeat
                    net=build(cell['model'],ctx,victim_seed)
                    if cell['defense']=='none':
                        ordinary(net,ctx)
                    elif cell['defense']=='asd':
                        train_asd(net,ctx['train_imgs'],ctx['train_labs'],
                                  torch.tensor(trusted,device=ctx['device']),directory/'asd')
                        assignments={}
                        for epoch,labeled_tensor in torch.load(directory/'asd'/'assignments.pt',weights_only=True).items():
                            labeled=labeled_tensor.numpy()
                            assignments[str(epoch)]=dict(poison_labeled_rate=float(labeled[pmask].mean()),
                                clean_unlabeled_rate=float((~labeled[~pmask]).mean()))
                        info['assignments']=assignments
                        defended_metrics=metrics(net,ctx)
                        del net
                        net=build(cell['model'],ctx,victim_seed)
                        asd_supervised(net,ctx,directory/'supervised_control')
                        info['matched_supervised_control']=metrics(net,ctx)
                        # Keep final output describing ASD, not the auxiliary control.
                        info['defended_metrics']=defended_metrics
                    else:
                        val=ctx['val_idx']
                        _,traces,validation=train_trajectory(net,ctx['train_imgs'],ctx['train_labs'],
                            ctx['test_imgs'][val],ctx['test_labs'][val],directory/'trajectory')
                        info['before_filter']=metrics(net,ctx)
                        suspicious,det=detect_telltale(traces,validation,ctx['device'],4200000+repeat,directory)
                        info.update(det)
                        info.update(detection_metrics(suspicious,pmask,ctx['train_labs'].cpu().numpy()))
                        # Preserve full mixed trajectories plus clean-control correspondence.
                        save_torch(directory/'loss_trajectories.pt',torch.from_numpy(traces))
                        kept=torch.tensor(np.flatnonzero(~suspicious),device=ctx['device'])
                        assert len(kept)>128
                        del net
                        net=build(cell['model'],ctx,victim_seed)
                        fu.train_from_scratch(net,ctx['train_imgs'][kept],ctx['train_labs'][kept],
                                             60,.1,125,[40],ctx['device'])
                    measured=info.pop('defended_metrics') if cell['defense']=='asd' else metrics(net,ctx)
                    control_kind='asd' if cell['defense']=='asd' else 'plain'
                    control_path=OUT/'controls'/cell['model']/f'{control_kind}_r{repeat}'/'result.json'
                    control_result=json.loads(control_path.read_text())
                    result=dict(version=VERSION,cell=cell,target=tidx,repeat=repeat,
                        success=int(measured['target_predictions'][str(tidx)]==5),
                        clean_accuracy=measured['clean_accuracy'],
                        clean_target_prediction=control_result['target_predictions'][str(tidx)],
                        clean_baseline_accuracy=control_result['clean_accuracy'],
                        num_poisons=len(idx),linf=float(poison['delta'].abs().max()),
                        seconds=time.monotonic()-begin,details=info,
                        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        gpu_peak_bytes=torch.cuda.max_memory_allocated())
                    atomic_json(directory/'result.json',result)
                    print('RESULT',cell_id,tidx,repeat,result['success'],result['seconds'],flush=True)
                    del net
                finally:
                    ctx['train_imgs'][idx]=clean
                gc.collect()
        del nets


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['check','pretrain','wo_pretrained','wo_ood','control','cell','aggregate'])
    parser.add_argument('--model',choices=MODELS,default=MODELS[0])
    parser.add_argument('--fraction',type=float,choices=[.2,.5],default=.2)
    parser.add_argument('--repeat',type=int,choices=REPEATS,default=0)
    parser.add_argument('--part',type=int)
    parser.add_argument('--cell',type=int)
    parser.add_argument('--kind',choices=['plain','asd','asd_supervised','telltale'],default='plain')
    args=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Run only inside a Slurm allocation'
    assert torch.cuda.is_available(), 'A full GPU allocation is required'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=True
    part=args.part if args.part is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    OUT.mkdir(parents=True,exist_ok=True)
    print('START',vars(args),torch.__version__,torch.cuda.get_device_name(),flush=True)
    if args.mode!='check':
        from protocol import source_fingerprint
        passed=json.loads((OUT/'checks_passed.json').read_text())
        assert passed['source_fingerprint']==source_fingerprint(), 'Source differs from validated snapshot'
    if args.mode=='aggregate':
        from aggregate import aggregate
        aggregate(OUT)
    elif args.mode=='check':
        from test_sweep import run_checks
        run_checks()
    elif args.mode=='pretrain':
        pretrain(args.model,args.fraction,args.repeat,part)
    elif args.mode=='control':
        control(args.model,part,args.kind)
    elif args.mode=='cell':
        run_cell(args.cell,part)
    else:
        from wicked import pretrained_scores, ood_scores
        _,ctx=context(args.model); fu.set_seed(42)
        directory=OUT/'wicked'; directory.mkdir(parents=True,exist_ok=True)
        if args.mode=='wo_pretrained':
            pretrained_scores(ctx,directory/'wo_pretrained.pt')
        else:
            ood_scores(ctx,ROOT/'data'/'tiny-imagenet-200.zip',ctx['val_idx'],directory)


if __name__=='__main__':
    main()
