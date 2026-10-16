# Stage R judge (machine output)

## vit_b_16: gate passed
- B0_transient: ["intrinsic", 1.0828180574789148, 1.0413556196192788]
- B4_density: ["partial", 0.5620827970968489, -0.5849625007211567]
- C_shortcut: ["mixed", ["mixed", "gated", "shortcut"]]
- B2_A_half_K2_vs_K0: [null, 1.0828180574789148]

## resnet50: gate passed
- B0_transient: ["intrinsic", 2.9010000604798694, 2.959468148620883]
- B4_density: ["density_insensitive", 0.19565384412075812, -0.2680264501201051]
- C_shortcut: ["shortcut", ["shortcut", "shortcut", "shortcut"]]
- B2_A_half_K2_vs_K0: [1.6265031912137706, 2.9010000604798694]

## qwen_squad: gate FAILED

## corpus (R1)
- qwen_squad: {"additive_share": 0.9693640206352143, "inherit_rho": 0.5479830634586206, "cross_seed_rho": 0.745609404808505, "leak_slope": 0.2776932043000673, "A1": "deleted", "stability": "stable", "inheritance": "middling", "leak": "partial", "cross_seed_pairs": 45, "schedule": "not_schedule", "schedule_count": "2/9"}
- llama: {"additive_share": 0.9595910607849578, "inherit_rho": 0.6939221132558187, "cross_seed_rho": 0.7257310522204086, "leak_slope": 0.013868733584189786, "A1": "deleted", "stability": "stable", "inheritance": "inherited", "leak": "specific", "cross_seed_pairs": 10, "schedule": "not_schedule", "schedule_count": "1/5"}
- resnet50: {"additive_share": 0.9683245666146908, "inherit_rho": 0.7235913906906958, "cross_seed_rho": 0.6762225171558792, "leak_slope": 0.3786892313011093, "A1": "deleted", "stability": "stable", "inheritance": "inherited", "leak": "partial", "cross_seed_pairs": 45}
- vit_b_16: {"additive_share": 0.9261011081907121, "inherit_rho": 0.48188493804174315, "cross_seed_rho": 0.732936435319258, "leak_slope": -0.12543407276451843, "A1": "deleted", "stability": "stable", "inheritance": "middling", "leak": "specific", "cross_seed_pairs": 45}
- vim_s: {"additive_share": 0.9326842526743127, "inherit_rho": 0.23642614477008053, "cross_seed_rho": 0.3617852590865206, "leak_slope": -0.6710240863865656, "A1": "deleted", "stability": "middling", "inheritance": "middling", "leak": "specific", "cross_seed_pairs": 10}
