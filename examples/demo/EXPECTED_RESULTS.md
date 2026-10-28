# Expected demo results

Output of `python run_demo.py` with MoCoP 1.0.0 on the shipped data (seed 42 split). Values may
differ in the third decimal across platforms and library versions.

## Trained edges

| mediator | kind | exposed (n) | estimate | 95% CI | max SMD after weighting | used |
|---|---|---|---|---|---|---|
| LAB_2160-0 | lab | 305 | -0.039 | -0.136, 0.058 | 0.032 | yes |
| LAB_2823-3 | lab | 298 | -0.009 | -0.047, 0.029 | 0.027 | yes |
| LAB_1988-5 | lab | 301 | -0.306 | -0.394, -0.219 | 0.044 | yes |
| LAB_718-7 | lab | 295 | 0.082 | -0.001, 0.165 | 0.037 | yes |
| EVENT | event | 354 | -0.293 (RR 0.746) | -0.663, 0.077 | 0.025 | yes |

True values (`truth.json`): CRP severity shift -0.367; event risk ratio 0.600.

## Effect on the outcome, all patients

| mode | estimand | risk unexposed | risk exposed | RR | 95% CI |
|---|---|---|---|---|---|
| curated | total | 0.235 | 0.212 | 0.899 | 0.868, 0.931 |
| curated | labs | 0.235 | 0.221 | 0.941 | 0.910, 0.972 |
| curated | event | 0.235 | 0.225 | 0.959 | 0.953, 0.965 |
| curated | event_held | 0.234 | 0.220 | 0.940 | 0.910, 0.972 |
| trained | total | 0.235 | 0.215 | 0.914 | 0.876, 0.954 |
| trained | labs | 0.235 | 0.221 | 0.941 | 0.908, 0.974 |
| trained | event | 0.235 | 0.229 | 0.974 | 0.946, 1.002 |
| trained | event_held | 0.234 | 0.220 | 0.940 | 0.907, 0.974 |

True total risk ratio: 0.791. The estimate is attenuated toward 1 because post-index labs are
represented by baseline values; direction and pathway decomposition are recovered.

## Outcome model

Test AUC 0.795, Brier score 0.141. Every directionality check passes.
