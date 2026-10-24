# External inputs, identities and rights

No raw melanoma input or state-bearing biological restart history is bundled. `manifests/external_inputs.json` lists six exact files: a binary view, gene manifest and controlled-model protocol for each of the 28-node and 70-node models. The scientific cases use different intervention semantics: `melanoma_xor` is the 28-node XOR-pirin case; `melanoma_persistent` is the 70-node force-pirin case.

Obtain the exact authorized inputs from the authors or a lawful original source under its conditions. An authoritative public download route for these exact derived views is not asserted here. Source access, permission and reproduction are separate questions. Stage files outside the companion:

```sh
python companion.py inputs --study melanoma_xor --source-dir ../authorized-files --input-dir ../permitted-inputs --acknowledge-local-use-rights
python companion.py inputs --study melanoma_xor --input-dir ../permitted-inputs --acknowledge-local-use-rights
```

The staging command requires each original basename in the source directory, validates size/SHA-256, and will not overwrite an existing destination. Acknowledgement means responsibility for lawful local use; it grants no redistribution license.

The 70-node available-workbook reconstruction differs from the retained primary binary view in **251 of 2,170 cells**. The exact retained identity is used where available; this package does not resolve its derivation or replace it with a reconstructed alternative. Without the exact permitted input, the affected new biological evaluations/training remain unavailable.

`manifests/continuation_external.json` identifies **113** additional external authorization, schema and restart files, covering **36 saved snapshots**. These include biological history and random-state data and are not packaged here. Their exact identities bind the historical continuation permission. The CLI cannot manufacture replacements, extend authorization or bypass the exact-environment gate.

The package does include learned policy arrays, scalar/aggregate records and count-only evaluation files. The 100 additional training-budget files retain only the `counts` array, with its seven integer columns described in `results/melanoma_training_budget/count_schema.json`. Full states, inputs, probability histories, replay buffers and biological RNG states were not copied from those source records. Removing state trajectories does not by itself confer rights over the derived outputs.

`manifests/evidence_sources.json` links numerical evidence to SHA-256 identities. A `sha256:` identifier binds original file bytes; a suffix selector addresses a value within the original file. A `manifest-sha256:` identifier binds an external directory's manifest. A count-only view is explicitly separate from the original raw file. A missing raw source is not falsely described as bundled.

Existing DDQN policy snapshots additionally retain optimizer moments and replay indices/priorities/positions. The replay metadata identifies positions in the external history; it does not contain the biological state vectors. These learned-state metadata remain subject to the same derived-output permission hold. Full restart histories required for live continuation remain external.
