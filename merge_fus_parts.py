#!/usr/bin/env python3
"""Validate and merge four target-partitioned FUS result directories."""

import argparse
import csv
import glob
import json
import math
import os
import shutil


def atomic_json(path, value):
    temporary = '%s.tmp.%d' % (path, os.getpid())
    with open(temporary, 'w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_copy(source, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = '%s.tmp.%d' % (destination, os.getpid())
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def population_std(values):
    if not values:
        return None
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def balanced_parts(values, count):
    quotient, remainder = divmod(len(values), count)
    parts = []
    for index in range(count):
        start = index * quotient + min(index, remainder)
        stop = start + quotient + (1 if index < remainder else 0)
        parts.append(values[start:stop])
    return parts


def summarize_overhead(records):
    def values(key):
        return [float(record[key]) for record in records
                if record.get(key) is not None]

    def total(key):
        found = values(key)
        return float(sum(found)) if found else None

    def mean(key):
        found = values(key)
        return float(sum(found) / len(found)) if found else None

    def maximum(key):
        found = values(key)
        return float(max(found)) if found else None

    return {
        'target_records': len(records),
        'selection_measurements': len(values('selection_wall_seconds')),
        'final_craft_measurements': len(values('final_craft_wall_seconds')),
        'selection_wall_seconds_total': total('selection_wall_seconds'),
        'selection_wall_seconds_mean_per_target': mean('selection_wall_seconds'),
        'selection_cuda_peak_allocated_bytes_max': maximum(
            'selection_cuda_peak_allocated_bytes'),
        'selection_cuda_incremental_peak_allocated_bytes_max': maximum(
            'selection_cuda_incremental_peak_allocated_bytes'),
        'selection_cuda_peak_reserved_bytes_max': maximum(
            'selection_cuda_peak_reserved_bytes'),
        'final_craft_wall_seconds_total': total('final_craft_wall_seconds'),
        'final_craft_wall_seconds_mean_per_target': mean('final_craft_wall_seconds'),
        'final_craft_cuda_peak_allocated_bytes_max': maximum(
            'final_craft_cuda_peak_allocated_bytes'),
        'final_craft_cuda_incremental_peak_allocated_bytes_max': maximum(
            'final_craft_cuda_incremental_peak_allocated_bytes'),
        'final_craft_cuda_peak_reserved_bytes_max': maximum(
            'final_craft_cuda_peak_reserved_bytes'),
        'process_max_rss_kb_max': maximum('process_max_rss_kb'),
        'records': records,
    }


def read_rows(path):
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def main(args):
    if os.path.basename(args.run_name) != args.run_name:
        raise SystemExit('--run-name must be one directory name')
    with open(args.target_file) as handle:
        targets = list(map(
            int, json.load(handle)['pairs'][args.class_pair]['indices']))
    if len(targets) != args.targets or len(set(targets)) != args.targets:
        raise SystemExit('target file must contain exactly %d unique targets'
                         % args.targets)

    run_dir = os.path.join(args.result_root, args.run_name)
    expected_parts = balanced_parts(targets, args.parts)
    all_rows = []
    summaries = []
    fieldnames = None
    part_manifest = []
    target_rank = {target: rank for rank, target in enumerate(targets)}

    for part_index, expected_targets in enumerate(expected_parts, start=1):
        label = 'part_%d_of_%d' % (part_index, args.parts)
        part_dir = os.path.join(run_dir, 'parts', label)
        result_path = os.path.join(part_dir, 'results.csv')
        summary_path = os.path.join(part_dir, 'summary.json')
        if not os.path.isfile(result_path) or not os.path.isfile(summary_path):
            raise SystemExit('incomplete %s: missing results.csv or summary.json'
                             % part_dir)
        current_fields, rows = read_rows(result_path)
        if fieldnames is None:
            fieldnames = current_fields
        elif current_fields != fieldnames:
            raise SystemExit('CSV columns differ in %s' % part_dir)
        pairs = [(int(row['target_idx']), int(row['victim_id'])) for row in rows]
        expected_pairs = {
            (target, victim)
            for target in expected_targets
            for victim in range(args.victims)
        }
        if len(rows) != len(expected_pairs) or set(pairs) != expected_pairs:
            raise SystemExit(
                '%s does not contain exactly %d target/victim evaluations'
                % (label, len(expected_pairs)))
        if len(set(pairs)) != len(pairs):
            raise SystemExit('%s contains duplicate target/victim rows' % label)
        if any(row['model'] != args.model for row in rows):
            raise SystemExit('%s contains the wrong model' % label)
        with open(summary_path) as handle:
            summaries.append(json.load(handle))
        all_rows.extend(rows)
        part_manifest.append({
            'part': part_index,
            'directory': os.path.relpath(part_dir, run_dir),
            'targets': expected_targets,
            'evaluations': len(rows),
            'gnu_time_files': [
                os.path.basename(path)
                for path in sorted(glob.glob(os.path.join(part_dir, 'job_gnu_time_*.txt')))
            ],
        })

        for target in expected_targets:
            for relative in (
                    os.path.join('poison_cache', 'delta_%d.pt' % target),
                    os.path.join('poison_cache', 'base_%d.json' % target),
                    os.path.join('overhead', 'target_%d.json' % target)):
                source = os.path.join(part_dir, relative)
                if not os.path.isfile(source):
                    raise SystemExit('missing required part artifact: ' + source)
                atomic_copy(source, os.path.join(run_dir, relative))

    all_pairs = [(int(row['target_idx']), int(row['victim_id'])) for row in all_rows]
    expected_all = {
        (target, victim) for target in targets for victim in range(args.victims)
    }
    if len(all_rows) != len(expected_all) or set(all_pairs) != expected_all:
        raise SystemExit('merged rows are not exactly the requested full evaluation')
    if len(set(all_pairs)) != len(all_pairs):
        raise SystemExit('duplicate target/victim rows across partitions')
    all_rows.sort(key=lambda row: (
        target_rank[int(row['target_idx'])], int(row['victim_id'])))

    results_path = os.path.join(run_dir, 'results.csv')
    temporary_results = '%s.tmp.%d' % (results_path, os.getpid())
    with open(temporary_results, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(temporary_results, results_path)

    overhead_records = []
    for target in targets:
        with open(os.path.join(run_dir, 'overhead', 'target_%d.json' % target)) as handle:
            overhead_records.append(json.load(handle))
    overhead = summarize_overhead(overhead_records)
    atomic_json(os.path.join(run_dir, 'overhead', 'summary.json'), overhead)

    per_target_rates = []
    for target in targets:
        successes = [
            int(row['success']) for row in all_rows
            if int(row['target_idx']) == target
        ]
        if len(successes) != args.victims:
            raise SystemExit('target %d lacks victim outcomes' % target)
        per_target_rates.append(sum(successes) / args.victims)
    clean_test_acc = [float(row['clean_test_acc']) for row in all_rows]

    summary = dict(summaries[0])
    summary.update({
        'num_targets': args.targets,
        'num_trials': len(all_rows),
        'asr_mean': sum(per_target_rates) / len(per_target_rates),
        'asr_std': population_std(per_target_rates),
        'cta_post_mean': sum(clean_test_acc) / len(clean_test_acc),
        'cta_post_std': population_std(clean_test_acc),
        'overhead_target_records': overhead['target_records'],
        'selection_overhead_measurements': overhead['selection_measurements'],
        'final_craft_overhead_measurements': overhead['final_craft_measurements'],
        'selection_wall_seconds_total': overhead['selection_wall_seconds_total'],
        'selection_wall_seconds_mean_per_target': overhead[
            'selection_wall_seconds_mean_per_target'],
        'selection_cuda_peak_allocated_bytes_max': overhead[
            'selection_cuda_peak_allocated_bytes_max'],
        'selection_cuda_incremental_peak_allocated_bytes_max': overhead[
            'selection_cuda_incremental_peak_allocated_bytes_max'],
        'final_craft_wall_seconds_total': overhead['final_craft_wall_seconds_total'],
        'final_craft_wall_seconds_mean_per_target': overhead[
            'final_craft_wall_seconds_mean_per_target'],
        'final_craft_cuda_peak_allocated_bytes_max': overhead[
            'final_craft_cuda_peak_allocated_bytes_max'],
        'final_craft_cuda_incremental_peak_allocated_bytes_max': overhead[
            'final_craft_cuda_incremental_peak_allocated_bytes_max'],
        'process_max_rss_kb_max': overhead['process_max_rss_kb_max'],
        'merged_from_target_parts': args.parts,
        'target_partition_sizes': [len(part) for part in expected_parts],
    })
    baseline = summary.get('cta_baseline_mean')
    summary['cta_drop_mean'] = (
        None if baseline is None else summary['cta_post_mean'] - float(baseline))
    tallies = [part.get('tally') for part in summaries]
    if tallies and all(isinstance(tally, list) for tally in tallies):
        width = len(tallies[0])
        if all(len(tally) == width for tally in tallies):
            summary['tally'] = [
                sum(int(tally[index]) for tally in tallies)
                for index in range(width)
            ]
    atomic_json(os.path.join(run_dir, 'summary.json'), summary)

    manifest = {
        'run_name': args.run_name,
        'model': args.model,
        'targets': targets,
        'victims_per_target': args.victims,
        'total_evaluations': len(all_rows),
        'parts': part_manifest,
    }
    atomic_json(os.path.join(run_dir, 'parts', 'merge_manifest.json'), manifest)

    combined_log = os.path.join(run_dir, 'parts_combined.log')
    temporary_log = '%s.tmp.%d' % (combined_log, os.getpid())
    with open(temporary_log, 'w') as output:
        for part in part_manifest:
            label = os.path.basename(part['directory'])
            output.write('===== %s =====\n' % label)
            path = os.path.join(run_dir, part['directory'], 'log.txt')
            if os.path.isfile(path):
                with open(path) as source:
                    shutil.copyfileobj(source, output)
            output.write('\n')
    os.replace(temporary_log, combined_log)

    print('merged %d FUS parts: %d targets x %d victims = %d evaluations'
          % (args.parts, args.targets, args.victims, len(all_rows)), flush=True)
    print('result:', results_path, flush=True)
    print('ASR: %.1f%%' % (100.0 * summary['asr_mean']), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--target-file', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--class-pair', default='dog-bird')
    parser.add_argument('--parts', type=int, default=4)
    parser.add_argument('--targets', type=int, default=10)
    parser.add_argument('--victims', type=int, default=6)
    args = parser.parse_args()
    if args.parts <= 0 or args.targets <= 0 or args.victims <= 0:
        parser.error('--parts/--targets/--victims must be positive')
    return args


if __name__ == '__main__':
    main(parse_args())
