#!/usr/bin/env python3
"""Normalize pasted run logs, update full.tex, and emit missing-run commands."""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

from process_full_results import (
    CELL_RE,
    EPSILONS,
    MODEL_LABELS,
    SUMMARY_RE,
    collect_table_values,
    populate_table,
)


HEADER_RE = re.compile(
    r'^--- (?P<select>ours|dpp) \| (?P<model>\S+) / (?P<attack>\S+) / '
    r'(?P<pair>\S+)(?: \| sharp .*?)? \| budget (?P<budget>[0-9.]+) ---$',
    re.MULTILINE,
)
RUN_START_RE = re.compile(r'=== run start: (?P<run>CIFAR10_\S+) on ')
RUN_SUMMARY_RE = re.compile(r'==== (?P<run>CIFAR10_\S+) : ASR = ')
TIMESTAMP_LINE_RE = re.compile(r'^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\]')

MODELS = ('ConvNetBN', 'ResNet20BN', 'VGG13BN')
ATTACKS = ('fc', 'gradmatch', 'sapa')
PAIRS = ('dog-bird', 'frog-airplane')
METHOD_OFFSETS = (('greedy', 1), ('greedy_J', 3), ('DPP_J', 4))
BUDGET_TEXT = {1: '0.001', 2: '0.002', 5: '0.005', 10: '0.01',
               20: '0.02', 40: '0.04'}
# Budget 0.05 was an accidental sweep value and is not part of full.tex.
# Drop matching run identities instead of retaining them as out-of-grid work.
DISCARDED_BUDGET_EPSILONS = {50}
# The 0.04 cells stay in full.tex, but the user does not want those runs in the
# generated work queue.
COMMAND_EXCLUDED_EPSILONS = {40}


def atomic_write(path, text):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    os.replace(str(tmp), str(path))


def split_fragments(text):
    """Return exact-run fragments plus blocks that never started a run."""
    headers = list(HEADER_RE.finditer(text))
    fragments = defaultdict(list)
    orphans = []
    if headers and text[:headers[0].start()].strip():
        orphans.append(text[:headers[0].start()].strip())
    elif not headers and text.strip():
        orphans.append(text.strip())

    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[header.start():end].rstrip()
        runs = set(RUN_START_RE.findall(block)) | set(RUN_SUMMARY_RE.findall(block))
        if not runs:
            orphans.append(block)
            continue
        if len(runs) != 1:
            raise ValueError('one pasted block contains multiple run identities: %r'
                             % sorted(runs))
        fragments[next(iter(runs))].append(block)
    return fragments, orphans, len(headers)


def canonical_block(run, fragments):
    """Merge replayed/resumed fragments into one compact chronological block."""
    header = fragments[0].splitlines()[0]
    seen = set()
    timestamped = []
    serial = 0
    for fragment in fragments:
        for line in fragment.splitlines()[1:]:
            if not TIMESTAMP_LINE_RE.match(line) or line in seen:
                continue
            seen.add(line)
            timestamped.append((line[1:20], serial, line))
            serial += 1
    timestamped.sort(key=lambda item: (item[0], item[1]))

    # A replay can contain the same final result under multiple timestamps. Keep
    # exactly the latest summary for this identity after checking its values agree.
    summary_lines = [line for _, _, line in timestamped
                     if '==== %s : ASR = ' % run in line]
    summary_values = set()
    for line in summary_lines:
        match = re.search(r'ASR = ([0-9.]+)%.*?\| CTA = ([0-9.]+)', line)
        if match:
            summary_values.add(match.groups())
    if len(summary_values) > 1:
        raise ValueError('conflicting summaries for %s: %r' % (run, summary_values))
    latest_summary = summary_lines[-1] if summary_lines else None

    lines = []
    for _, _, line in timestamped:
        if '==== %s : ASR = ' % run in line and line != latest_summary:
            continue
        lines.append(line)
    return header + '\n' + '\n'.join(lines).rstrip() + '\n\n'


def parse_run_key(run):
    match = re.match(
        r'^CIFAR10_(ConvNetBN|ResNet20BN|VGG13BN)_'
        r'(fc|gradmatch|sapa)_ours_(dog-bird|frog-airplane)_b([0-9.]+)_', run)
    if not match:
        return None
    model, attack, pair, budget = match.groups()
    jacobian = '_jacw' in run
    dpp = '_seldpp' in run
    if dpp and jacobian:
        method = 'DPP_J'
    elif not dpp and jacobian:
        method = 'greedy_J'
    elif not dpp and not jacobian:
        method = 'greedy'
    else:
        return None
    return model, attack, pair, round(float(budget) * 1000), method


def table_cells(tex):
    lines = tex.splitlines()
    cells_by_row = {}
    for model, label in MODEL_LABELS.items():
        marker = '%% %s' % label
        section = lines.index(marker)
        for epsilon in EPSILONS:
            row = next(i for i in range(section, len(lines))
                       if lines[i] == '& %d & ' % epsilon)
            cells = CELL_RE.findall(lines[row + 1] + lines[row + 2])
            if len(cells) != 30:
                raise ValueError('%s epsilon %d has %d cells, expected 30'
                                 % (label, epsilon, len(cells)))
            cells_by_row[(model, epsilon)] = cells
    return cells_by_row


def command_for(model, attack, pair, epsilon, method):
    fields = []
    if method == 'greedy':
        fields.append('USE_JACOBIAN_SCORE=0')
    else:
        fields.extend(('USE_JACOBIAN_SCORE=1', 'JACOBIAN_WEIGHT=1.0',
                       'JACOBIAN_BATCH_SIZE=64'))
    fields.extend(('CLASS_PAIR="%s"' % pair, 'MODEL="%s"' % model,
                   'ATTACK="%s"' % attack))
    if attack == 'sapa':
        fields.extend(('SHARP_MODE="worst"', 'SHARP_SIGMA="0.05"'))
    fields.append('BUDGETS="%s"' % BUDGET_TEXT[epsilon])
    fields.append('SELECT="%s"' % ('dpp' if method == 'DPP_J' else 'ours'))
    if method == 'DPP_J':
        fields.append('SEL_ALPHA=2.0')
    return ' '.join(fields) + ' sh sel_dpp.sh'


def missing_commands(tex, incomplete_keys, excluded_epsilons=()):
    cells = table_cells(tex)
    missing = []
    for model in MODELS:
        for pair_index, pair in enumerate(PAIRS):
            for attack_index, attack in enumerate(ATTACKS):
                start = pair_index * 15 + attack_index * 5
                for epsilon in EPSILONS:
                    if epsilon in excluded_epsilons:
                        continue
                    row = cells[(model, epsilon)]
                    for method, offset in METHOD_OFFSETS:
                        if row[start + offset] != '--':
                            continue
                        key = (model, attack, pair, epsilon, method)
                        missing.append((key not in incomplete_keys, model, pair,
                                        attack, epsilon, method,
                                        command_for(model, attack, pair, epsilon,
                                                    method)))
    # Existing interrupted work comes first; within each section keep a stable grid order.
    missing.sort(key=lambda item: item[:-1])
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', default='results/main_result.txt')
    parser.add_argument(
        '--source', action='append', default=[],
        help='input log to import; repeat to consolidate multiple legacy logs')
    parser.add_argument('--tex', default='full.tex')
    parser.add_argument('--commands', default='remaining_full_tex_commands.txt')
    parser.add_argument('--report', default='result_hygiene_report.txt')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    main_path, tex_path = map(Path, (args.main, args.tex))
    command_path, report_path = map(Path, (args.commands, args.report))
    source_paths = [Path(path) for path in args.source] or [main_path]
    main_raw = '\n\n'.join(path.read_text().rstrip() for path in source_paths
                            if path.exists())
    fragments, orphans, header_count = split_fragments(main_raw)
    discarded_runs = {
        run for run in fragments
        if (parse_run_key(run) is not None
            and parse_run_key(run)[3] in DISCARDED_BUDGET_EPSILONS)
    }
    fragments = {
        run: parts for run, parts in fragments.items() if run not in discarded_runs
    }

    completed = {run for run, parts in fragments.items()
                 if any(RUN_SUMMARY_RE.search(part) for part in parts)}
    incomplete = set(fragments) - completed
    ordered_runs = [run for run in fragments if run in completed]
    ordered_runs.extend(run for run in fragments if run in incomplete)
    new_main = ''.join(canonical_block(run, fragments[run])
                       for run in ordered_runs).lstrip()
    values = collect_table_values(new_main)
    new_tex, table_updates = populate_table(tex_path.read_text(), values)
    incomplete_keys = {key for key in (parse_run_key(run) for run in incomplete)
                       if key is not None and key[3] in EPSILONS}
    all_commands = missing_commands(new_tex, incomplete_keys)
    commands = missing_commands(new_tex, incomplete_keys,
                                COMMAND_EXCLUDED_EPSILONS)
    omitted_commands = len(all_commands) - len(commands)
    deferred_commands = [
        item for item in commands
        if item[3] in ('gradmatch', 'sapa') and item[4] in (20, 40)
    ]
    regular_commands = [item for item in commands if item not in deferred_commands]
    command_sections = [
        '\n'.join(item[-1] for item in section)
        for section in (regular_commands, deferred_commands) if section
    ]
    command_text = '\n\n'.join(command_sections) + ('\n' if command_sections else '')
    resume_count = sum(not item[0] for item in commands)

    outside_grid = sorted(
        run for run in incomplete
        if (parse_run_key(run) is not None and parse_run_key(run)[3] not in EPSILONS))
    duplicate_identities = sum(len(parts) > 1 for parts in fragments.values())
    fragment_count = sum(len(parts) for parts in fragments.values())
    report = [
        'raw headers: %d' % header_count,
        'identified fragments: %d' % fragment_count,
        'unique run identities: %d' % len(fragments),
        'discarded accidental budget-0.05 identities: %d' % len(discarded_runs),
        'identities with duplicate/resume fragments: %d' % duplicate_identities,
        'completed identities retained: %d' % len(completed),
        'incomplete identities retained: %d' % len(incomplete),
        'header-only/lock-skip fragments ignored: %d' % len(orphans),
        'full.tex cells updated: %d' % table_updates,
        'remaining table commands: %d' % len(commands),
        'blank budget-0.04 cells intentionally omitted from commands: %d'
        % omitted_commands,
        'commands that resume an identified incomplete run: %d' % resume_count,
        'commands for table cells with no incomplete fragment: %d'
        % (len(commands) - resume_count),
        'incomplete runs outside the table budget grid: %d' % len(outside_grid),
    ]
    if discarded_runs:
        report.extend(['', 'Discarded accidental budget-0.05 runs:']
                      + sorted(discarded_runs))
    if outside_grid:
        report.extend(['', 'Excluded incomplete runs outside full.tex:'] + outside_grid)
    report_text = '\n'.join(report) + '\n'

    print(report_text, end='')
    if not args.dry_run:
        main_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(main_path, new_main)
        atomic_write(tex_path, new_tex)
        atomic_write(command_path, command_text)
        atomic_write(report_path, report_text)


if __name__ == '__main__':
    main()
