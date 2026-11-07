"""Tiny selection and exact checkpoint checks; run inside the allocated job."""
import copy
import os
from pathlib import Path
import tempfile

import torch

import final_update as fu
import worker


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this check inside a Slurm allocation.')
    torch.set_num_threads(1)
    labels = torch.tensor([2, 5, 0, 5] + [5] * 40)
    first = worker.choose_bases(labels, 'first_m', 8, 843, 0)
    assert torch.equal(first, (labels == 5).nonzero(as_tuple=True)[0][:8])
    draw1 = worker.choose_bases(labels, 'random', 8, 843, 1)
    assert torch.equal(draw1, worker.choose_bases(labels, 'random', 8, 843, 1))
    assert not torch.equal(draw1, worker.choose_bases(labels, 'random', 8, 843, 2))
    fu.set_seed(123)
    images, classes = torch.randn(12, 3), torch.arange(12) % 2
    settings = dict(epochs=12, learning_rate=0.1, momentum=0.9, weight_decay=0,
                    batch_size=4, decay=[8])
    fu.set_seed(456)
    reference = torch.nn.Linear(3, 2)
    fu.train_from_scratch(reference, images, classes, 12, 0.1, 4, [8], 'cpu')
    fu.set_seed(456)
    resumed = torch.nn.Linear(3, 2)
    with tempfile.TemporaryDirectory(dir=os.environ['SLURM_TMPDIR']) as temporary:
        directory = Path(temporary)
        normal = worker.stopping
        worker.stopping = lambda: (directory / 'training.pt').exists()
        try:
            worker.fit_victim(resumed, images, classes, settings, directory, dict(target=843, victim=0))
            raise AssertionError('The victim did not pause.')
        except worker.Paused:
            pass
        finally:
            worker.stopping = normal
        worker.fit_victim(resumed, images, classes, settings, directory, dict(target=843, victim=0))
    assert all(torch.equal(value, resumed.state_dict()[key])
               for key, value in reference.state_dict().items())

    fu.set_seed(789)
    nets = [torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))]
    bases, target = torch.rand(3, 3, 2, 2), torch.rand(3, 2, 2)
    arguments = (nets, bases, target, 1, lambda x: x, 0.03, 0.004, 4, 2, 'cpu')
    fu.set_seed(42)
    expected, objective = fu.craft_gradmatch(*arguments)
    saved = {}

    def pause(progress):
        if progress['restart'] == 0 and progress['next_step'] == 2:
            saved.update(copy.deepcopy(progress))
            raise worker.Paused('Synthetic crafting pause.')

    fu.set_seed(42)
    try:
        fu.craft_gradmatch(*arguments, checkpoint_callback=pause)
        raise AssertionError('Crafting did not pause.')
    except worker.Paused:
        pass
    actual, actual_objective = fu.craft_gradmatch(*arguments, resume_state=saved)
    assert torch.equal(expected, actual) and objective == actual_objective
    print('PASS: deterministic first-m/random draws; exact SGD and crafting resume.', flush=True)


if __name__ == '__main__':
    main()
