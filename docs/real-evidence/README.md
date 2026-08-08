# Real-data evidence

This directory separates reproducible evidence from the synthetic demo.

## NOAA Storm Events 2024

The 40-row corpus is sampled deterministically from the official NOAA NCEI
Storm Events details file. NOAA narratives are non-sensitive US federal
government records. The source hash, sampling rule, public-domain status, and
corpus hash are recorded in `noaa_storm_events_2024_provenance.json`; the corpus
validator report has zero errors and zero warnings.

Two isolated DeepSeek V4 Flash sessions coded the corpus without seeing each
other's labels. Both files have zero schema or evidence-span errors. Strict
free-form construct matching produces 24 shared edges out of 87 unique edges,
Jaccard `0.275862`, and union-universe kappa `-0.565553`. This is evidence that
the current exact-name agreement unit over-penalizes naming granularity. It is
not human gold and it does not validate model extraction accuracy.

## OSMI Mental Health in Tech 2014

The 40-comment OSMI subset is real, anonymous survey text selected
deterministically from the pinned Figshare file. The Figshare description says
CC BY-SA 4.0 while its API license field says CC BY 4.0; this repo follows the
more restrictive BY-SA terms. The corpus passes validation with zero warnings.

Because the comments contain mental-health disclosures, they were not sent to
an external DeepSeek service. No coder agreement or extraction-quality claim is
made for this corpus.
