"""Small mathematical checks plus allocated-GPU integration/calibration."""
import inspect
import time
import os
import resource
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from protocol import cells, attack_cell, subsets, atomic_json, MODELS, TARGETS
from defenses import (sce_loss, interleave, split_asd, meta_parameters,
                      mixmatch_epoch, train_asd, train_trajectory, detect_telltale,
                      global_normalize, suspect_clusters)
from wicked import knn_outlier


def unit_checks():
    grid=cells()
    assert len(grid)==120 and len({tuple(c.items()) for c in grid})==120
    assert sum(c['defense']=='none' for c in grid)==84
    assert sum(c['fraction']!=1 or c['pool']!=1 for c in grid)==48
    assert sum(c['selector'].startswith('wo_') for c in grid)==12
    for c in grid:
        assert attack_cell(c)['defense']=='none'
    labels=np.repeat(np.arange(10),5000)
    prior=None
    for r in range(3):
        trusted,visible,candidates=subsets(labels,r)
        assert len(trusted)==100 and len(np.unique(trusted))==100
        assert len(visible[.2])==10000 and len(visible[.5])==25000
        assert set(visible[.2])<set(visible[.5])<set(visible[1.])
        for f in (1.,.5,.2):
            assert set(candidates[f])<=set(visible[f])
            assert not set(candidates[f])&set(trusted)
            assert len(candidates[f])>=250
        if prior is not None:
            assert not np.array_equal(prior,visible[.2])
        prior=visible[.2]
    logits=torch.randn(100,10); y=torch.arange(100)%10
    reference=.1*F.cross_entropy(logits,y,reduction='none')
    reference+=(-np.log(1e-4))*(1-logits.softmax(1).gather(1,y[:,None]).squeeze())
    torch.testing.assert_close(sce_loss(logits,y),reference)
    chunks=[torch.arange(i*64,(i+1)*64) for i in range(3)]
    assert all(torch.equal(a,b) for a,b in zip(chunks,interleave(interleave(chunks))))
    y=torch.arange(1000)%10; score=torch.arange(1000).float()
    trusted=torch.arange(100)
    assert int(split_asd(score,y,trusted,0).sum())==100
    assert int(split_asd(score,y,trusted,5).sum())==200
    assert int(split_asd(score,y,trusted,60).sum())==500
    assert torch.equal(split_asd(score,y,trusted,90,score*2),score>=500)
    sample=torch.tensor([[1.,0.],[1.,.01],[1.,-.01],[-1.,0.]])
    assert int(knn_outlier(sample,k=2).argmax())==3
    z=np.random.default_rng(42).normal(size=(50,10)); p=np.arange(50)[::-1]
    np.testing.assert_allclose(global_normalize(z)[p],global_normalize(z[p]),atol=1e-10)
    assert suspect_clusters(np.array([3,3,3,7,-1])).tolist()==[False,False,False,True,True]
    assert 'poison' not in str(inspect.signature(detect_telltale))
    # Dropping a component must preserve per-surrogate standardization.
    import final_update as fu
    d,m=torch.randn(25),torch.randn(25)
    torch.testing.assert_close(fu.combine_distance_margin(d,m,1,0),fu.standardize(m))
    torch.testing.assert_close(fu.combine_distance_margin(d,m,1,1),fu.standardize(d))
    torch.testing.assert_close(2*fu.combine_distance_margin(d,m,1,.5),fu.combine_distance_margin(d,m,1))


def run_checks():
    import final_update as fu
    from run import context, build, load_pool, OUT
    import fcntl, subprocess
    gate=(OUT/'validation.lock').open('w')
    fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
    current=os.environ['SLURM_JOB_ID']
    atomic_json(OUT/'validation_owner.json',dict(job=current,account=os.environ.get('SLURM_JOB_ACCOUNT')))
    jobs_file=OUT/'validation_jobs.txt'
    if jobs_file.exists():
        other=[j for j in jobs_file.read_text().split() if j.isdigit() and j!=current]
        if other:
            subprocess.run(['scancel',*other],check=False,timeout=30)
    unit_checks()
    from defenses import augment
    assert augment(torch.randn(4,3,32,32)).is_contiguous()
    measurements={}
    for model in MODELS:
        args,ctx=context(model)
        assert ctx['train_imgs'].shape==(50000,3,32,32)
        assert len(ctx['val_idx'])==1000 and not set(TARGETS)&set(ctx['val_idx'].cpu().tolist())
        trusted,_,candidates=subsets(ctx['train_labs'].cpu().numpy(),0)
        nets=load_pool(model,1.,0,ctx)
        begin=time.monotonic()
        candidate=torch.tensor(candidates[.2],device=ctx['device'])
        labels=torch.full_like(ctx['train_labs'],-1); labels[candidate]=5
        selected=fu.select_base_ours(nets,ctx['train_imgs'],labels,ctx['test_imgs'][TARGETS[0]],
            5,50,1.,ctx['device'],base_dist='cosine',distance_margin_coef=.5)
        assert set(selected.cpu().tolist())<=set(candidate.cpu().tolist())
        score_seconds=time.monotonic()-begin
        begin=time.monotonic()
        base,adv,_=fu.craft_selected_bases(args,ctx,nets[:5],selected,
                                         ctx['test_imgs'][TARGETS[0]],5,steps=3,restarts=1)
        assert torch.isfinite(adv).all() and float((adv-base).abs().max())<=8/255+1e-6
        del nets,base,adv
        net=build(model,ctx,4200000)
        begin=time.monotonic()
        fu.train_from_scratch(net,ctx['train_imgs'],ctx['train_labs'],2,.1,128,[],ctx['device'])
        torch.cuda.synchronize(); epoch_seconds=(time.monotonic()-begin)/2
        meta_parameters(net)
        assert any(p.requires_grad for p in net.classifier.parameters())
        assert any(not p.requires_grad for p in net.parameters())
        del net
        net=build(model,ctx,4200000)
        directory=OUT/'calibration'/os.environ['SLURM_JOB_ID']/model
        begin=time.monotonic()
        train_asd(net,ctx['train_imgs'],ctx['train_labs'],
                  torch.tensor(trusted,device=ctx['device']),directory,epochs=1,steps=20)
        torch.cuda.synchronize(); asd_smoke_seconds=time.monotonic()-begin
        begin=time.monotonic()
        mask=torch.zeros_like(ctx['train_labs'],dtype=torch.bool); mask[:25000]=True
        opt=torch.optim.Adam(net.parameters(),lr=.002)
        mixmatch_epoch(net,opt,ctx['train_imgs'],ctx['train_labs'],mask,60,steps=50)
        torch.cuda.synchronize(); asd_step_seconds=(time.monotonic()-begin)/50
        del net,opt
        net=build(model,ctx,4200000)
        val=ctx['val_idx']; begin=time.monotonic()
        _,trace,validation=train_trajectory(net,ctx['train_imgs'],ctx['train_labs'],
            ctx['test_imgs'][val],ctx['test_labs'][val],directory/'trajectory',epochs=2)
        torch.cuda.synchronize(); trajectory_epoch_seconds=(time.monotonic()-begin)/2
        assert trace.shape==(50000,2) and np.isfinite(trace).all()
        measurements[model]=dict(victim_epoch_seconds=epoch_seconds,selector_seconds=score_seconds,
            asd_step_seconds=asd_step_seconds,asd_smoke_seconds=asd_smoke_seconds,
            trajectory_epoch_seconds=trajectory_epoch_seconds,
            max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print('CALIBRATION',model,measurements[model],flush=True)
        del net,ctx
        torch.cuda.empty_cache()
    # Exercise the complete blind detector on synthetic trajectories, never used as research results.
    rng=np.random.default_rng(314)
    trace=np.abs(rng.normal(size=(50000,120))).astype('float32')
    directory=OUT/'calibration'/os.environ['SLURM_JOB_ID']/'detector'
    directory.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    suspicious,info=detect_telltale(trace,np.ones(120),'cuda:0',42,directory)
    assert suspicious.shape==(50000,)
    measurements['detector_50000_seconds']=time.monotonic()-begin
    measurements['max_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert measurements['max_rss_kb']<6*1024*1024
    atomic_json(OUT/'calibration.json',measurements)
    from protocol import source_fingerprint
    atomic_json(OUT/'checks_passed.json',dict(job=os.environ['SLURM_JOB_ID'],source_fingerprint=source_fingerprint(),checks='math + both architectures + GM + ASD + trajectories + blind detector'))
    print('ALL CHECKS PASSED',flush=True)


def resume_checks():
    """Exercise ASD's late-stage meta update after resuming a checkpoint."""
    from run import context, build, OUT
    from defenses import save_torch, rng_state
    for model in MODELS:
        _,ctx=context(model)
        net=build(model,ctx,4200000)
        trusted,_,_=subsets(ctx['train_labs'].cpu().numpy(),0)
        directory=OUT/'calibration'/os.environ['SLURM_JOB_ID']/model/'resume'
        opt=torch.optim.Adam(net.parameters(),lr=.002)
        save_torch(directory/'checkpoint.pt',dict(net=net.state_dict(),opt=opt.state_dict(),
            epoch=90,history=[],assignments={60:torch.zeros(50000,dtype=torch.bool)},
            score_traces={},rng=rng_state()))
        train_asd(net,ctx['train_imgs'],ctx['train_labs'],
                  torch.tensor(trusted,device=ctx['device']),directory,epochs=91,steps=2)
        assignments=torch.load(directory/'assignments.pt',weights_only=True)
        assert all(t.device.type=='cpu' for t in assignments.values())
        assert int(assignments[91].sum())==25000
        del net,opt,ctx
        torch.cuda.empty_cache()
    atomic_json(OUT/'resume_checks_passed.json',dict(job=os.environ['SLURM_JOB_ID']))
    print('ASD RESUME AND META-STAGE CHECKS PASSED',flush=True)
