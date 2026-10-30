# Supplied SharedCNNComponents snapshot

This directory preserves all seven files supplied in `SharedCNNComponents`.
The `source_manifest.json` maps their original paths to these stored copies and
records SHA-256 hashes. Python files and the original README have a `.txt`
suffix so they do not become another active implementation or collected test suite.
Their contents are unchanged.

The package's original README describes an alternative portrait resize and
augmentation policy and earlier APIs. Its relative links refer to its intended
original repository layout. Use [the integration guide](../../SHARED_COMPONENTS_INTEGRATION.md)
for supported commands, approved settings and the mapping to the current API.
Do not install this directory's historical requirements over the tested project
environment or replace the active `resnet18/common.py` with the reference copy.
