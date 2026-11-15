First video attempt (2026-11-14 ~12:47). NOT valid examples: the recorder ran torch with the default
thread count while independent validation ran with 1 thread. The batch-1 policy output then differs by
~2.4e-7 and contact-rich episodes diverge, so several re-runs did not reproduce the validation outcome
(see manifest.csv, column reproduced). File names carry the VALIDATION outcome, not the re-run outcome.
Recorder fixed to 1 thread; valid videos are in the parent folder.
