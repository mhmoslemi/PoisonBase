// Freeze only the ten missing Rand/R/M cells; leave existing batches untouched.
const fs=require('fs'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const root=path.resolve(__dirname,'../..'),batch=path.join(root,'sbatch/rebuttal_missing_cells_vulcan');
const sourceDir=path.join(root,'sbatch/rebuttal_basis_resnet_fix_vulcan');
const sourcePlanFile=path.join(root,'sbatch/rebuttal_basis_lambda_killarney/plan.json');
const sourcePlan=JSON.parse(fs.readFileSync(sourcePlanFile)),sha=b=>crypto.createHash('sha256').update(b).digest('hex');
assert(!fs.existsSync(batch),'Batch already prepared; do not overwrite a submitted snapshot.');
const cells=[],tasks=[];
for(const [row,components]of [['c100_sea_gm_b001',['R','M']],['c100_wardrobe_sapa_b002',['R','M']],['tiny_gm_b005',['random','R','M']],['tiny_sapa_b005',['random','R','M']]]){
    const source=sourcePlan.cells.find(c=>c.row===row&&c.lambda===.1);assert(source);
    for(const component of components){
        const c=JSON.parse(JSON.stringify(source));c.id=row+'_'+component;c.component=component;c.site='vulcan';delete c.lambda;
        c.source_cell=source.id;c.args.lambda_margin=1;c.args.base=component==='random'?'random':'ours';
        assert.equal(c.targets.length,4);assert.equal(c.args.num_victims,4);assert.equal(c.args.victim_epochs,50);
        cells.push(c);const tiny=c.args.dataset==='TinyImageNet';
        const common={cell:c.id,site:'vulcan',dataset:c.args.dataset,memory:tiny?'18G':'7G',account:tiny?'aip-bghaddar':'aip-yiweilu'};
        if(!tiny)tasks.push({...common,id:c.id,stage:'all',targets:c.targets,seconds:7200,requires:[]});
        else for(const target of c.targets){const craft=c.id+'_craft_t'+target;tasks.push({...common,id:craft,stage:'craft',targets:[target],seconds:18000,requires:[]});tasks.push({...common,id:c.id+'_victims_t'+target,stage:'victims',targets:[target],seconds:10800,requires:[craft]});}
    }
}
assert.equal(cells.length,10);assert.equal(tasks.length,52);
const plan={purpose:'Only ten previously unsubmitted Rand/R/M cells in tab:reb-c-scaling',results:'rebuttal_missing_cells_result',source_plan:'sbatch/rebuttal_basis_lambda_killarney/plan.json',source_plan_sha256:sha(fs.readFileSync(sourcePlanFile)),cells,tasks};
fs.mkdirSync(batch+'/code/experiments/rebuttal_mr',{recursive:true});
for(const file of ['final_update.py','networks.py','utils.py'])fs.copyFileSync(sourceDir+'/code/'+file,batch+'/code/'+file);
let worker=fs.readFileSync(sourceDir+'/code/experiments/rebuttal_mr/run.py','utf8');
const start=worker.indexOf('def select('),end=worker.indexOf('\n\ndef poison(',start);assert(start>=0&&end>start);
const selection=String.raw`def select(cell, ctx, nets, target):
    args = cell['args']
    seed = args['seed'] * 100003 + target
    fu.set_seed(seed)
    count = round(args['budget'] * len(ctx['train_labs']))
    if cell['component'] == 'random':
        generator = torch.Generator(device='cpu').manual_seed(seed)
        indices = fu.select_base_random(ctx['train_labs'].cpu(), ctx['y_adv'], count,
                                       'cpu', generator).to(ctx['device'])
    else:
        common = dict(base_dist=args['base_dist'],
                      distance_margin_coef=1.0 if cell['component'] == 'R' else 0.0)
        values = (nets, ctx['train_imgs'], ctx['train_labs'], ctx['test_imgs'][target],
                  ctx['y_adv'], count, args['lambda_margin'], ctx['device'])
        if args.get('sel_mode'):
            indices = fu.select_base_ours_div(*values, mode=args['sel_mode'],
                alpha=args['sel_alpha'], pool=args['sel_pool'], mu=args['sel_mu'], **common)
        else:
            indices = fu.select_base_ours(*values, **common)
    if len(indices) != count or len(indices.unique()) != count:
        raise RuntimeError('Wrong poison count or duplicate bases.')
    if not bool((ctx['train_labs'][indices] == ctx['y_adv']).all()):
        raise RuntimeError('Selected bases have the wrong class.')
    return indices
`;
worker=worker.slice(0,start)+selection+worker.slice(end);
worker=worker.replace('BASIS margin-weight cells; reuse the validated M/R checkpointed runner.','Missing Rand/R/M cells; reuse the validated checkpointed runner.').replaceAll('submit_basis_resnet_fix_vulcan_ghaddar.sh','submit_missing_cells_vulcan.sh');
fs.writeFileSync(batch+'/code/experiments/rebuttal_mr/run.py',worker);
fs.mkdirSync(batch+'/code/experiments/rebuttal_missing_cells',{recursive:true});
fs.copyFileSync(__dirname+'/check_selection.py',batch+'/code/experiments/rebuttal_missing_cells/check_selection.py');
let job=fs.readFileSync(root+'/sbatch/rebuttal_basis_lambda_extra_vulcan/job.sh','utf8');
job=job.replaceAll('submit_basis_lambda_remaining_vulcan.sh','submit_missing_cells_vulcan.sh');
job=job.replace('cd "$ROOT"\nsrun','cd "$ROOT"\npython -u "$BATCH/code/experiments/rebuttal_missing_cells/check_selection.py" "$BATCH"\nsrun');
assert(job.includes('check_selection.py'));
fs.writeFileSync(batch+'/job.sh',job);
let submit=fs.readFileSync(root+'/submit_basis_resnet_fix_vulcan_ghaddar.sh','utf8');
submit=submit.replaceAll('rebuttal_basis_resnet_fix_vulcan','rebuttal_missing_cells_vulcan').replaceAll('rebuttal_basis_resnet_fix_result','rebuttal_missing_cells_result');
submit=submit.replace('aip-bghaddar|aip-boyuwang','aip-bghaddar|aip-yiweilu').replaceAll('bl_$task','mc_$task').replace('2ghaddar','missing-cells');
submit=submit.replace('# Vulcan replacement: two corrected Ghaddar jobs, maximum six hours each.','# Ten missing cells: Tiny on Ghaddar, CIFAR-100 on Lu.');
submit=submit.replace('Two corrected ResNet20 jobs, both on Ghaddar.','52 jobs: 48 Tiny on Ghaddar and four CIFAR-100 on Lu.');
fs.writeFileSync(root+'/submit_missing_cells_vulcan.sh',submit);
fs.writeFileSync(batch+'/plan.json',JSON.stringify(plan,null,2)+'\n');
for(const file of ['data.tsv','surrogates.tsv']){const lines=fs.readFileSync(root+'/sbatch/rebuttal_basis_lambda_killarney/'+file,'utf8').trim().split('\n');fs.writeFileSync(batch+'/'+file,[lines[0],...lines.slice(1).filter(l=>['CIFAR100','TinyImageNet'].includes(l.split('\t')[2]))].join('\n')+'\n');}
const files=['plan.json','job.sh','data.tsv','surrogates.tsv','code/final_update.py','code/networks.py','code/utils.py','code/experiments/rebuttal_mr/run.py','code/experiments/rebuttal_missing_cells/check_selection.py'];
fs.writeFileSync(batch+'/code.sha256',files.map(f=>sha(fs.readFileSync(batch+'/'+f))+'  '+f).join('\n')+'\n');
console.log('Prepared 10 cells, 40 poison sets, 160 victim trainings, 52 resumable jobs.');
