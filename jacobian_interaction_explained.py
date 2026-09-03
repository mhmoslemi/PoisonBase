"""
ANNOTATED COPY of ``_backbone_gradient_interactions`` from final_update.py.

This file is documentation, not production code.  The function body below is a
verbatim copy of the one used by ``--use_jacobian_score`` (the ``ours`` and
``dpp`` selectors) and by the A-containing component selectors.  Only comments
have been added.  If you edit the real one, this copy goes stale.

=============================================================================
WHAT IS BEING COMPUTED
=============================================================================

For every candidate image x_i in the poison class, we want one scalar

    A_i  =  < grad_phi ell_i ,  grad_phi L_adv,t >                        (*)

where

    phi         = every parameter of the network EXCEPT the final linear head
                  (.classifier).  "Backbone" = feature extractor.
    ell_i       = CE( f_phi(x_i), y_adv )      loss of candidate i
    L_adv,t     = CE( f_phi(x_t), y_adv )      loss of the TARGET image x_t
                  relabeled to the adversarial class
    < . , . >   = flat dot product over all of phi, i.e. sum over every
                  parameter tensor, summed elementwise:
                      sum_p  (d ell_i / d phi_p) * (d L_adv,t / d phi_p)

INTERPRETATION.  grad_phi L_adv,t is the direction in weight space that most
increases the target's adversarial loss; minus it is the direction that makes
the victim classify the target as y_adv.  A_i is large and positive when a
gradient step on candidate i pushes the weights along that same direction.  So
A_i measures "does training on this candidate drag the model toward
misclassifying the target the way we want".  That is why the selector prefers
large A_i.

WHY IT IS CALLED A "JACOBIAN" SCORE.  Stack the per-candidate losses of a batch
into a vector

    F(phi) = [ ell_1(phi), ell_2(phi), ..., ell_B(phi) ]     (B outputs)

Its derivative is the Jacobian J, a  B x |phi|  matrix whose row i is exactly
grad_phi ell_i.  Then the whole batch of (*) is a single matrix-vector product:

    A_batch = J @ g_t,        g_t = grad_phi L_adv,t                     (**)

We never build J.  We only ever need J times a known vector, and that is
precisely what a forward-mode JVP gives us in ONE forward pass.

COST.  Naive way: one backward pass per candidate  ->  |pool| backward passes,
i.e. thousands per surrogate per target.  This way: 1 backward pass for g_t,
then 1 JVP per batch of 64.  Roughly a 64x saving, and no  |pool| x |phi|
tensor is ever allocated (that matrix would be tens of GB).

=============================================================================
THE TWO BACKENDS
=============================================================================

Backend 1 (primary) -- forward-mode AD:
    torch.func.jvp(F, (phi,), (g_t,))  returns  (F(phi), J @ g_t).
    Exactly (**).  One forward pass, dual numbers carried along.

Backend 2 (fallback) -- dummy-weight double backward.  Used only if some
    operator in the model has no forward-AD rule.  Introduce w in R^B, all
    zeros, requires_grad.  Define

        S(w, phi) = w . [ell_1, ..., ell_B] = sum_i w_i ell_i

    First differentiate w.r.t. phi, keeping the graph:

        grad_phi S = sum_i w_i * grad_phi ell_i                           (1)

    Dot that with the (detached, constant) g_t:

        D(w) = < grad_phi S , g_t > = sum_i w_i * < grad_phi ell_i, g_t >
             = sum_i w_i * A_i                                            (2)

    D is LINEAR in w, so differentiating once more w.r.t. w reads off the
    coefficients:

        d D / d w_i = A_i                                                 (3)

    Same numbers as backend 1, two reverse passes instead of one forward pass.
    Note the value of w never matters (it is zero) -- only its position in the
    graph.  This is the standard trick for getting per-sample gradient
    quantities without a per-sample loop.

=============================================================================
WHY eval() IS LOAD-BEARING, NOT COSMETIC
=============================================================================

The batching above is only valid if row i of J is candidate i's OWN gradient.
In BatchNorm training mode the batch mean/variance couple all images in a
chunk, so ell_i would depend on x_j for j != i and row i would be polluted.
``core.eval()`` switches BN to its running statistics, which are constants
here, so the losses are independent and

    A_i computed with batch=64   ==   A_i computed with batch=1

exactly.  ``jacobian_batch_size`` is therefore a pure memory knob and can never
change a run's result.  The train/eval flags are captured before and restored
in the ``finally`` block so scoring cannot leak state into training.

=============================================================================
WHERE THE RESULT GOES
=============================================================================

In ``_ours_pointwise_score`` the returned vector becomes, per surrogate:

    cost_i = standardize(d_i) + lambda * standardize(M_i)
                             - beta * standardize(A_i)

    d_i = feature distance to target      (small = good)
    M_i = adversarial logit margin        (small = good)
    A_i = this function                   (LARGE = good, hence the MINUS)

standardize is (v - mean) / std over the candidate pool.  It is needed because
the raw dot products in (*) have an arbitrary scale that differs by orders of
magnitude between independently trained surrogates; without it, one surrogate's
gradient magnitude would dominate the ensemble average.  Costs are averaged
over surrogates, then the selector takes the LOWEST costs (``ours``) or feeds
q_i = exp(-alpha * cost_i) to the greedy DPP (``dpp``).

The sibling function ``_full_gradient_interactions`` is this same routine with
phi = ALL parameters including the head; it backs ``--sel_exact_alignment``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _restore_training_states(states):
    """Restore per-module train/eval flags without changing any other state."""
    for module, training in states:
        module.training = training


def _log_jacobian_backend(backend, reason=None):
    """Stub for this standalone copy; the real one logs once per process."""
    if reason:
        print('  Jacobian score exact backend: %s (functional JVP unavailable: %s)'
              % (backend, reason))
    else:
        print('  Jacobian score exact backend: %s' % backend)


def _backbone_gradient_interactions(net, candidates, x_t_norm, y_adv,
                                    batch_size=64):
    """Return exact candidate/target CE-gradient dots over non-head parameters.

    Returns a tensor of shape (len(candidates),) whose entry i is A_i from (*),
    plus the name of the backend that produced it.

    Args:
        net:        a trained surrogate.  Must expose its final linear layer as
                    ``.classifier`` so the head can be excluded from phi.
        candidates: (N, C, H, W) NORMALIZED images, the whole poison-class pool.
        x_t_norm:   the single normalized target image (3-D or 4-D both work).
        y_adv:      the adversarial class index, used as the label for BOTH the
                    candidates (which really are that class) and the target
                    (which is not -- that is the whole point of the attack).
        batch_size: candidate chunk size.  Memory only; see the eval() note.
    """
    # ---------------------------------------------------------------------
    # 0. Sanity checks.  Fail here rather than after hours of surrogate
    #    training.
    # ---------------------------------------------------------------------
    if batch_size <= 0:
        raise ValueError('jacobian batch size must be positive, got %d' % batch_size)

    # Unwrap DataParallel so named_parameters() keys match what functional_call
    # expects (no 'module.' prefix).
    core = net.module if isinstance(net, nn.DataParallel) else net

    # The split into phi (backbone) vs head is defined by this attribute, so a
    # model that does not expose it cannot be scored.  Silently guessing which
    # layer is the head would change the meaning of A_i.
    if not hasattr(core, 'classifier'):
        raise ValueError('Jacobian scoring requires the model to expose its final '
                         'linear head as .classifier; got %s' % type(core).__name__)
    if not isinstance(core.classifier, nn.Linear):
        raise ValueError('Jacobian scoring requires .classifier to be nn.Linear; got %s'
                         % type(core.classifier).__name__)

    # ---------------------------------------------------------------------
    # 1. Split the parameters:   phi = backbone (differentiated)
    #                            head = .classifier (frozen constant)
    #
    #    Compared by id(), not by name, so a parameter shared between the head
    #    and the body could never end up in both dicts.
    #
    #    Excluding the head is a modelling choice: the head is retrained from
    #    scratch by the victim, so alignment in the last layer is not
    #    transferable, while backbone/feature-extractor alignment is.
    # ---------------------------------------------------------------------
    named_params = dict(core.named_parameters())
    head_ids = {id(p) for p in core.classifier.parameters()}
    backbone = {name: p for name, p in named_params.items() if id(p) not in head_ids}
    fixed = {name: p for name, p in named_params.items() if id(p) in head_ids}
    if not backbone:
        raise ValueError('Jacobian scoring found no parameters outside .classifier')

    # Buffers = BatchNorm running_mean / running_var / num_batches_tracked.
    # They are NOT parameters, they are constants here, and they are what
    # eval() makes the network use instead of per-batch statistics.
    buffers = dict(core.named_buffers())

    # ---------------------------------------------------------------------
    # 2. Force eval mode, remembering the previous flags.
    #    THIS IS THE CORRECTNESS-CRITICAL LINE, not a performance tweak.
    #    See the eval() section of the module docstring: it is what makes
    #    row i of the Jacobian equal candidate i's own gradient.
    # ---------------------------------------------------------------------
    states = [(module, module.training) for module in core.modules()]
    core.eval()

    # Accept the target as (C,H,W) or (1,C,H,W); the network needs a batch dim.
    target = x_t_norm.unsqueeze(0) if x_t_norm.ndim == candidates.ndim - 1 else x_t_norm

    # The target is labeled y_adv, i.e. the class we WANT it misclassified as.
    # This is what makes L_adv,t an "adversarial" loss rather than the target's
    # ordinary training loss.
    label_target = torch.full((target.shape[0],), int(y_adv), dtype=torch.long,
                              device=target.device)

    # ---------------------------------------------------------------------
    # 3. Turn the network into a pure function of phi.
    #
    #    Normally net(x) reads its weights from inside the module, which makes
    #    it opaque to torch.func.  functional_call runs the SAME module but
    #    with the weights supplied explicitly, so
    #
    #        phi  |-->  logits
    #
    #    becomes an ordinary mathematical function we can differentiate,
    #    JVP through, or evaluate at perturbed weights.  The head params in
    #    ``fixed`` are merged in unchanged, so they contribute no derivative.
    # ---------------------------------------------------------------------
    def functional_logits(backbone_params, x):
        params = dict(fixed)
        params.update(backbone_params)
        return torch.func.functional_call(core, (params, buffers), (x,))

    target_grad = None      # will hold g_t
    jvp_error = None        # non-None iff we fell back to backend 2
    try:
        # The caller (_ours_pointwise_score) is decorated @torch.no_grad, so
        # grad tracking must be switched back on explicitly here.
        with torch.enable_grad():
            try:
                # ---------------------------------------------------------
                # BACKEND 1: forward-mode JVP.
                # ---------------------------------------------------------
                # Old torch builds lack these; check up front so we land in
                # the fallback with a clean message instead of an AttributeError
                # halfway through a batch loop.
                if not all(hasattr(torch.func, name)
                           for name in ('functional_call', 'grad', 'jvp')):
                    raise RuntimeError('torch.func functional_call/grad/jvp is unavailable')

                # --- 3a. g_t = grad_phi L_adv,t -----------------------------
                # ONE ordinary reverse-mode pass.  This is the only full
                # parameter-gradient object the function ever materializes.
                # Shape: same pytree/shapes as ``backbone``.
                def target_loss(backbone_params):
                    return F.cross_entropy(functional_logits(backbone_params, target),
                                           label_target)

                target_grad = torch.func.grad(target_loss)(backbone)
                # Detach: g_t is a fixed DIRECTION from here on, never a thing
                # we backprop through.
                target_grad = {name: grad.detach() for name, grad in target_grad.items()}

                # --- 3b. one JVP per candidate chunk ------------------------
                values = []
                for start in range(0, len(candidates), batch_size):
                    batch = candidates[start:start + batch_size]

                    # F(phi) = [ell_1, ..., ell_B].  reduction='none' is what
                    # keeps the B losses separate; with the default 'mean' we
                    # would get (1/B) * sum_i A_i, a single number, not the
                    # per-candidate vector we need.
                    def candidate_losses(backbone_params):
                        logits = functional_logits(backbone_params, batch)
                        # Candidates genuinely belong to y_adv, so this is
                        # their ordinary training loss.
                        labels = torch.full((len(batch),), int(y_adv), dtype=torch.long,
                                            device=batch.device)
                        return F.cross_entropy(logits, labels, reduction='none')

                    # THE KEY LINE.
                    #   jvp(f, primals, tangents) -> (f(primals), Jf @ tangents)
                    # with primals = phi and tangents = g_t, the second output is
                    #
                    #   tangent[i] = sum_p (d ell_i / d phi_p) * (g_t)_p
                    #              = < grad_phi ell_i , grad_phi L_adv,t >
                    #              = A_i                                  (**)
                    #
                    # All B values from ONE forward pass; J is never built.
                    _, tangent = torch.func.jvp(candidate_losses, (backbone,),
                                                (target_grad,))
                    values.append(tangent.detach())
                result = torch.cat(values)          # shape (len(candidates),)
                backend = 'torch.func JVP'

            except Exception as exc:
                # ---------------------------------------------------------
                # BACKEND 2: dummy-weight double backward.
                # Reached when an operator has no forward-AD rule.
                # ---------------------------------------------------------
                jvp_error = exc
                # NOTE: ``values`` from the aborted JVP loop is deliberately
                # abandoned and EVERY batch is recomputed below, so a single
                # run never mixes results from two backends.  They agree
                # analytically, but not necessarily bit-for-bit, and a mixed
                # vector would be neither.
                fallback_backbone = {
                    name: value.detach().requires_grad_(True)
                    for name, value in backbone.items()
                }

                def fallback_logits(x):
                    params = dict(fixed)
                    params.update(fallback_backbone)
                    return torch.func.functional_call(core, (params, buffers), (x,))

                values = []
                try:
                    # g_t again, this time with plain autograd.  Guarded by
                    # ``is None`` because the JVP path may already have
                    # succeeded at 3a and failed only later, in which case that
                    # g_t is reused.
                    if target_grad is None:
                        loss_t = F.cross_entropy(fallback_logits(target), label_target)
                        grads_t = torch.autograd.grad(
                            loss_t, tuple(fallback_backbone.values()))
                        target_grad = {
                            name: grad.detach()
                            for name, grad in zip(fallback_backbone, grads_t)
                        }
                    backbone_values = tuple(fallback_backbone.values())
                    target_values = tuple(target_grad.values())

                    for start in range(0, len(candidates), batch_size):
                        batch = candidates[start:start + batch_size]

                        # w in R^B, all zeros.  Its VALUE is irrelevant (see
                        # eq. (3): the answer is the coefficient of w_i, not a
                        # function of w).  What matters is that it sits in the
                        # graph so we can differentiate with respect to it.
                        weights = torch.zeros(len(batch), device=batch.device,
                                              dtype=batch.dtype, requires_grad=True)
                        labels = torch.full((len(batch),), int(y_adv), dtype=torch.long,
                                            device=batch.device)
                        losses = F.cross_entropy(
                            fallback_logits(batch), labels, reduction='none')

                        # eq. (1):  mixed = grad_phi ( sum_i w_i ell_i )
                        #                 = sum_i w_i * grad_phi ell_i
                        # create_graph=True keeps w's dependence alive for the
                        # second differentiation.
                        mixed = torch.autograd.grad(
                            torch.dot(weights, losses), backbone_values,
                            create_graph=True)

                        # eq. (2):  directional = < mixed , g_t >
                        #                       = sum_i w_i * A_i
                        # A scalar, linear in w.  The elementwise product and
                        # .sum() per tensor is the flat dot product over phi.
                        directional = sum((g * v).sum()
                                          for g, v in zip(mixed, target_values))

                        # eq. (3):  d directional / d w_i = A_i
                        # Reading the coefficients straight off a linear form.
                        interaction = torch.autograd.grad(directional, weights)[0]
                        values.append(interaction.detach())

                    result = torch.cat(values)
                    backend = 'dummy-weight double backward'

                except Exception as fallback_error:
                    # Both exact routes failed.  Raise with BOTH causes rather
                    # than silently substituting an approximation, which would
                    # quietly change what the paper's A_i means.
                    raise RuntimeError(
                        'exact Jacobian scoring failed with both batched backends; '
                        'torch.func JVP error: %s; double-backward error: %s'
                        % (jvp_error, fallback_error)) from fallback_error
    finally:
        # Always put the module's train/eval flags back, even on the error
        # paths, so scoring can never leave the surrogate in eval mode.
        _restore_training_states(states)

    # Log which backend actually ran (once per process in the real file), so a
    # finished run records how its A_i was produced.
    reason = None
    if jvp_error is not None:
        reason = '%s: %s' % (type(jvp_error).__name__, str(jvp_error).split('\n')[0])
    _log_jacobian_backend(backend, reason)

    # result[i] = A_i, one scalar per candidate, in the candidate pool's order.
    return result, backend
