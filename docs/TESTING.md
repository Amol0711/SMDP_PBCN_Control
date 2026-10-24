# Testing boundaries and recorded outcomes

## Recorded numerical validation

The installed-environment validation on 22 October 2026 ran `python -B test.py` successfully: **104 distinct tests**, comprising **82 numerical/portability contracts and 22 structural tests**. The 22 structural tests were also run separately with `python -I -S tests/test_export_structure.py`; that repetition is not added to the distinct total. See [the validation report](VALIDATION.md) for the command coverage, comparison criteria, exactness exceptions and limitations.

The validated numerical archive has SHA-256 `bba4521505f3fe8a3e13df201405ee8b8f7d4a01968f4773d42aca7b18fb92dc`. This documentation-synchronized distribution has a different archive identity. Its executable code, tests, configurations, data and plotting payload match the validated version; `manifests/validated_payload.json` records those unchanged identities. The full numerical suite was not repeated merely to update prose. This binding does not certify arbitrary future code changes.

## Structural checks

The standard-library export suite checks inventory handling, unsafe paths, altered/missing/extra files, symlinks, external output isolation, syntax, data-dependency routing and fail-closed guard order with a deliberately mocked mismatch. It does not run a scientific model or test the actual host's numerical eligibility. The numerical suite requires the separately specified dependencies. A test pass is not a proof of the paper's assumptions or a statement about every biological execution path.

## Repeating validation

First authenticate the ZIP externally, extract it into a new directory and verify the manifest. Run each applicable command into its own new external output directory. Preserve stdout, stderr, return code, effective environment, exact input identities and the archive hash. Failed and interrupted output is not accepted evidence. Keep blocked commands separate from successes. Run the unchanged actual eligibility check before any biological continuation. An expected rejection demonstrates enforcement, not biological reproduction.

## Installation and execution limits

A genuinely isolated virtual environment was created without inherited system/user numerical packages. Installation of the exact core requirements failed at package-index DNS resolution; the subsequent witness command failed because NumPy was absent. **No clean numerical installation passed**, and dependency incompatibility was not established. Do not vendor or copy the host's installed packages to simulate isolation. The biological extra installation was not attempted.

The actual historical-environment check rejected Linux 6.18.55 against the locked 6.18.44 descriptor despite a matching recorded CPU and numerical packages. No descriptor was edited, and no override was introduced. No exact melanoma inputs or biological restart histories were staged. Permission-only rejections are not input, state-restoration or model-validation passes. Later checks must report their own actual environment rather than borrowing this verdict.
