"""UMPAv2 training infrastructure.

Sub-modules:
    - config_v2:           Training configs (phase-based + SAM3-style)
    - phase_scheduler_v2:  3-phase freeze/unfreeze scheduler
    - trainer_v2:          V1-style trainer (Method 1)
    - trainer_v2_sam3:     SAM3-style trainer (Method 2a)
    - trainer_v2_allperm:  All-permutation trainer (Method 2b)
    - evaluate_v2:         Evaluation for UMPAv2 models
"""
