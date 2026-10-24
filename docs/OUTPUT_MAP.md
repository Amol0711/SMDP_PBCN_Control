# Numerical output map

This map links article and guide item identifiers to computational outputs. Article theory, proof, conceptual diagrams, comparison prose and pseudocode are not included. Table CSVs contain displayed values; linked numerical records retain unrounded values.

| Item | Study | Operation |
|---|---|---|
| article.figure.1 | Noncomputational | Not bundled; see the article. |
| article.figure.2 | four_state_reset | `python -B companion.py figures --only dwell_mechanism --out ../article_figure_2` |
| article.figure.3 | four_state_continuing | `python -B companion.py figures --only continuing_diagnostics --out ../article_figure_3` |
| guide.figure.C4 | melanoma_persistent | `python -B companion.py figures --only melanoma70_endpoints --out ../guide_figure_C4` |
| article.table.1 | Noncomputational | Not bundled; see the article. |
| article.table.2 | four_state_reset | `python -B companion.py export-output --id article.table.2 --out ../article_table_2` |
| article.table.3 | melanoma_xor | `python -B companion.py export-output --id article.table.3 --out ../article_table_3` |
| article.table.4 | melanoma_persistent | `python -B companion.py export-output --id article.table.4 --out ../article_table_4` |
| article.table.5 | lactose_first_arrival | `python -B companion.py export-output --id article.table.5 --out ../article_table_5` |
| guide.figure.C1 | lactose_meanfield | `python -B companion.py figures --only lactose_meanfield --out ../guide_figure_C1` |
| guide.figure.C2 | melanoma_xor | `python -B companion.py figures --only melanoma28_endpoints --out ../guide_figure_C2` |
| guide.figure.C3 | melanoma_xor | `python -B companion.py figures --only melanoma28_learning --out ../guide_figure_C3` |
| guide.table.C1 | Noncomputational | `python -B companion.py export-output --id guide.table.C1 --out ../guide_table_C1` |
| guide.table.C2 | lactose_first_arrival, lactose_meanfield | `python -B companion.py export-output --id guide.table.C2 --out ../guide_table_C2` |
| guide.table.C3 | melanoma_xor, melanoma_persistent, lactose_first_arrival, four_state_reset | `python -B companion.py export-output --id guide.table.C3 --out ../guide_table_C3` |
| guide.table.C4 | four_state_reset | `python -B companion.py export-output --id guide.table.C4 --out ../guide_table_C4` |
| guide.table.C5 | four_state_reset | `python -B companion.py export-output --id guide.table.C5 --out ../guide_table_C5` |
| guide.table.C6 | four_state_continuing | `python -B companion.py export-output --id guide.table.C6 --out ../guide_table_C6` |
| guide.table.C7 | melanoma_training_budget | `python -B companion.py export-output --id guide.table.C7 --out ../guide_table_C7` |
| guide.table.C8 | four_state_reset | `python -B companion.py export-output --id guide.table.C8 --out ../guide_table_C8` |
| guide.table.C9 | melanoma_xor | `python -B companion.py export-output --id guide.table.C9 --out ../guide_table_C9` |
| article.algorithm.1 | four_state_reset, four_state_continuing | Not bundled; see the article. |
| guide.table.C10 | melanoma_persistent | `python -B companion.py export-output --id guide.table.C10 --out ../guide_table_C10` |

The complete map is in `manifests/paper_outputs.json`. Export copies saved values and is distinct from reaggregation, policy evaluation, training or fresh optimization. Numerical plotted coordinates are unchanged.
