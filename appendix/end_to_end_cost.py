"""Lower half of tab:computational-cost: per-target attack-construction time.

Poison optimization is identical under Random and DPP, so the only difference is
the selection stage. Craft time is read from the ConvNetBN logs already on disk;
selection time comes from appendix/out/cost_ConvNetBN.json, written by
profile_selection.py. Nothing is re-run.
"""
import glob, json, os, re, statistics

sel = {}
p = 'appendix/out/cost_ConvNetBN.json'
if os.path.exists(p):
    sel = {r['selection']: r['total'] for r in json.load(open(p))['rows']}
else:
    print('  (run the profiler first -- selection times unknown, showing craft only)')

print('  %-6s %12s %12s %12s   %s' % ('attack', 'Random s', 'DPP s', 'overhead %', 'craft'))
for att, tag in (('FC', 'fc'), ('GM', 'gradmatch'), ('SAPA', 'sapa')):
    secs = []
    for f in glob.glob('ours_result/CIFAR10_ConvNetBN_%s_*_b0.005_*/log.txt' % tag):
        secs += [int(x) for x in re.findall(r'crafted \d+ poisons in (\d+) s', open(f).read())]
    if not secs:
        print('  %-6s   no ConvNetBN b0.005 craft on disk yet' % att)
        continue
    craft = statistics.median(secs)
    r = craft + sel.get('Random', 0.0)
    d = craft + sel.get('DPP', 0.0)
    over = 100.0 * (d - r) / r if r else float('nan')
    print('  %-6s %12.1f %12.1f %12.1f   %.1f s/target (n=%d)' % (att, r, d, over, craft, len(secs)))
