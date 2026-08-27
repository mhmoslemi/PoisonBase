# Residual-Suppression Validation

## Objective

Validate the mechanism behind the proposition without claiming that the experiment proves it. The proposition states that a large positive signed margin \(M_i\) makes the adversarial-label residual \(\|r_i\|_2\) small, which in turn suppresses both components of the local gradient alignment. Here, \(M_i\) is the margin, not the residual.

## Primary candidate-level experiment

For a fixed target and each candidate, save

- the signed margin \(M_i\);
- the residual norm \(\|r_i\|_2\);
- the classifier-head interaction
  \[
  H_i=(r_i^\top r_t)(h_i^\top h_t);
  \]
- the backbone interaction
  \[
  A_i=\langle J_i^\top W^\top r_i,J_t^\top W^\top r_t\rangle.
  \]

Pool candidates only after computing these quantities separately for every target and model. Within each candidate pool, divide examples into deciles according to \(M_i\). For every decile, report the median and 90th percentile of \(\|r_i\|_2\), \(|H_i|\), and \(|A_i|\). Plot these quantities against the margin decile and optionally overlay the upper bounds from the proposition.

The expected result is that the high-margin deciles have smaller residual norms and smaller upper tails for both alignment terms. This directly tests the proposed residual-suppression mechanism. It does not require either alignment term to be positive and does not predict attack success by itself.

## Small selector ablation

If additional poisoning runs are affordable, compare the following Greedy selectors using the same targets, seeds, poison budget, and downstream attack.

1. **Full score** selects by \(\widetilde R_i-\gamma\widetilde M_i+\beta\widetilde A_i\).
2. **No-margin score** selects by \(\widetilde R_i+\beta\widetilde A_i\).
3. **High-margin control** first restricts candidates to the highest-margin quartile and then ranks them by \(\widetilde R_i+\beta\widetilde A_i\).

The high-margin control is the key intervention. It deliberately chooses candidates with favorable feature and Jacobian scores from a residual-suppressed region. Report the selected candidates' mean margin, residual norm, \(|H_i|\), \(|A_i|\), and ASR. The proposition predicts weaker local alignment for this control, but it does not guarantee lower ASR after full poison optimization and victim retraining.

## Recommended scope

Run the candidate-level analysis first because it requires no victim retraining. If \(M_i\), \(r_i\), \(H_i\), and \(A_i\) are already cached, the analysis and plots can run on CPU. Otherwise, use one GPU pass to compute and save the exact Jacobian interactions. For the optional ASR study, use one architecture, one class pair, a low poison budget, and GM and SAPA. DPP is unnecessary because this experiment isolates the pointwise score and the proposition.


