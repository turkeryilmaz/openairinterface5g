# Memory-profiler semantic catalog v1

This directory is the version-controlled source closure for the OAI
memory-lifetime archive verifier. It contains the canonical byte-level schema
definitions, semantic catalogs, deterministic validators, and focused unit
tests needed by `oai_memprof_build_evidence.py`,
`oai_memprof_archive_composer.py`, and `oai_memprof_softmodem_launcher.py`.

The files were promoted from the publication campaign's ignored staging area
because entry points under `tools/profiling/memory/` must remain runnable from
a fresh Git clone. Campaign evidence, validation transcripts, Python bytecode,
and other generated state are deliberately excluded. Future schema revisions
must use a new versioned catalog rather than silently changing the meaning of
an existing archived object.
