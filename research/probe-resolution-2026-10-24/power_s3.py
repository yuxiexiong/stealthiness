"""The arithmetic behind NEXT_STEPS.zh-CN.md section 1.2: how many seeds per dose cell.

  python3 -I power_s3.py

Binary read-out, so a dose cell is a proportion.  Two questions:
  - with k seeds per cell, can the most extreme table even reach significance?
  - how many seeds does a given pair of true proportions need at power 0.8?
Nothing here is specific to this project; it is the power calculation that
poc/contracts/README.md requires of any main criterion with a sampling distribution.
"""
from math import comb, sqrt

Z_A, Z_B = 1.959963985, 0.8416212336          # alpha 0.05 two-sided, power 0.80


def fisher_two_sided(a, b, c, d):
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def p(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    obs = p(a)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs + 1e-12)


def n_per_group(p1, p2):
    return (Z_A + Z_B) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2


print("most extreme table at k seeds per cell (0/k vs k/k):")
for k in (2, 3, 4, 5, 6):
    p = fisher_two_sided(0, k, k, 0)
    print(f"  k={k}  Fisher two-sided p = {p:.4f}   {'significant' if p < 0.05 else 'NOT significant'}")

print("\nobserved 1/4 vs 3/4 (what a 0.25/0.75 truth gives in expectation at k=4):")
print(f"  Fisher two-sided p = {fisher_two_sided(1, 3, 3, 1):.4f}")

print("\nseeds per cell at power 0.80:")
for pair in ((0.1, 0.9), (0.2, 0.8), (0.25, 0.75), (0.3, 0.7), (0.4, 0.6)):
    print(f"  {pair[0]} vs {pair[1]}: {n_per_group(*pair):.1f}"
          f"  (normal approximation; at this n also check the exact test)")

print("\nreading: k=4 works only if the ladder brackets a near-step transition.  A gradual"
      "\ntransition in dose needs roughly 12 to 21 seeds per cell, so the ladder should lose"
      "\ncells before it loses k.")
