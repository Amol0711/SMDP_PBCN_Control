# PD-SMAC computational companion

Implementations, fixed numerical records, evaluation procedures and native plotting workflows for **Update-Rate-Constrained Semi-Markov Actor–Critic Learning for Probabilistic Boolean Control Networks**, by Jiahao Wu, Amol Yerudkar and Carmen Del Vecchio.

This package contains only computational implementations, data, numerical plots and supporting instructions. It excludes the manuscript, theoretical results and proofs, conceptual figures, reviewer correspondence and private execution records. It contains no Git history. Author permission, project licensing and rights to redistribute derived outputs must be resolved before public sharing. No license is granted by this package; read [access and licensing](ACCESS_AND_LICENSE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Start with saved evidence

Run commands from this directory. The following use only the Python standard library and do not train or evaluate a policy:

```sh
python companion.py verify
python companion.py studies
python companion.py outputs
python companion.py export-output --id article.figure.2 --out ../dwell-figure-evidence
```

Each output directory must be new and outside the companion. `export-output` copies the exact saved dependencies; it does **not** recalculate statistics or certify a new experiment. The [output map](docs/OUTPUT_MAP.md) identifies all three figures, five tables and one algorithm in the article, and all four figures and ten tables in the computational guide. Noncomputational article items are classified explicitly rather than exported as numerical artifacts.

The [22-page computational guide](docs/computational_guide.pdf) explains the benchmarks, intervention rules, metrics, statistical conventions and evidence. Its computational-documentation source and numerical tables are included. Article tables are supplied as plain data, not manuscript LaTeX fragments. The [single command reference](docs/COMMANDS.md) distinguishes saved-artifact export, aggregation, policy evaluation, new training, certificate checking and optimization.

## Scientific studies

The seven descriptive study identifiers are `four_state_reset`, `four_state_continuing`, `melanoma_xor`, `melanoma_persistent`, `melanoma_training_budget`, `lactose_meanfield` and `lactose_first_arrival`. They share one implementation and command interface. Original mathematical model identifiers are preserved internally so that protocol and policy ownership remain traceable.

Reset-trained, continuing and first-arrival objectives are not interchangeable. In particular, the continuing stochastic cohort remains infeasible at its recorded endpoint; a feasible same-family witness is not a learned optimum; and first-arrival greedification can destroy feasibility. The [limitations](docs/LIMITATIONS.md) and guide retain adverse findings.

## Layout

| Directory | Content |
|---|---|
| `pd_smac/` | Shared numerical components and study-specific routines |
| `studies/` | Scientific case navigation, without duplicate implementations |
| `results/` | Fixed policies, scalar records, exact-reference material and count-only evidence |
| `figures/` | One current set of six native numerical figures and their 29 CSV dependencies |
| `docs/` | Computational guide, commands, output routes, metrics, access and coverage |
| `manifests/` | Transport-independent scientific identities, external-input fingerprints and output maps |
| `environment/` | Dependency specifications and environment distinctions |
| `tests/` | Structural export checks and numerical contract tests |

There are no bundled melanoma input tables or state-bearing biological restart trajectories. The 100 count-only long-training files retain scalar evaluation counts, not the underlying state/input paths. These derived records and the other learned outputs still require rights approval before redistribution. Exact biological inputs and all 113 restart/control/schema files remain external and hash-gated.

## Reproduction limits

The applicable numerical commands were independently validated on 22 October 2026 in the installed environment. The [validation report](docs/VALIDATION.md) records 104 distinct passing tests, exact saved-policy and continuing-state comparisons, scoped aggregation, reference and software-restart checks, and the blocked paths. This distribution updates computational documentation, output routing and the structural route-count check. Numerical implementations, configurations, saved policies and underlying numerical records are unchanged and bound by [payload hashes](manifests/validated_payload.json). The earlier numerical execution is not being claimed as a fresh run of this new archive.

Transport verification is not numerical reproduction. The historical biological-continuation guard rejected the recorded host, and the genuinely isolated installation failed at package-index DNS resolution before NumPy installation. No clean numerical installation, new melanoma execution, fresh constrained first-arrival optimization or public-release permission is claimed.

The historical biological continuation guard requires the exact recorded environment and original authorization/input identities. A mismatch stops execution before a biological model or state is constructed. There is no override flag. The guard does not authorize new seeds or new experiments.

No repository URL, release DOI or license has been invented. Citation metadata is supplied in `CITATION.cff`. The associated-article entry uses its current title and authors; software credits retain the supplied attribution and should be confirmed separately. Add the actual approved release identity only after publication authorization and verification.
