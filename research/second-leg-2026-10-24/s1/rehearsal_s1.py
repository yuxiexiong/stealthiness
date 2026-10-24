"""CONTRACT_S1 offline rehearsal. Every criterion is demonstrated BOTH ways (it can be
met and it can fail), plus the degenerate branches. Exit 0 only if all demos behave as
declared. This is synthetic data only; the real-data gate is gate_s1.py."""
import sys
sys.path.insert(0, ".")
import s1

G = list(range(260, 530, 10))
ok = []


def check(name, got, want):
    good = got == want
    ok.append(good)
    print(("  PASS  " if good else "  FAIL  ") + name + f"   got={got} want={want}")


print("C1 headroom: flip band in cells, threshold NO_HEADROOM_CELLS =", s1.NO_HEADROOM_CELLS)
# satisfiable: two arms 140 steps apart
check("C1 can be met (band 14 > 1)", s1.flip_band([330, 470], G)["cells"] > s1.NO_HEADROOM_CELLS, True)
# not satisfiable: two arms one cell apart
check("C1 can fail (band 1 <= 1)", s1.flip_band([330, 340], G)["cells"] > s1.NO_HEADROOM_CELLS, False)
# degenerate: identical arms -> no band at all, must not crash and must read as no headroom
check("C1 degenerate all-equal -> None", s1.flip_band([330, 330], G), None)
# degenerate: an arm that never crosses on the axis
check("C1 one arm never installs", s1.flip_band([330, None], G)["cells"], 20)

print("C2 contrast: order span vs init span")
check("C2 can be met (14 vs 3)", s1.span([330, 470])["cells"] > s1.span([330, 360])["cells"], True)
check("C2 can fail (3 vs 14)", s1.span([330, 360])["cells"] > s1.span([330, 470])["cells"], False)
check("C2 degenerate single point -> None", s1.span([330]), None)

print("C3 conservation and mediation")
# clean_acc spread
check("C3 spread small can be met", max([.62, .64, .63]) - min([.62, .64, .63]) <= 0.05, True)
check("C3 spread large can fail", max([.62, .30, .63]) - min([.62, .30, .63]) <= 0.05, False)
# mediation: perfectly monotone -> small p; scrambled -> large p
rho_a, p_a = s1.perm_p_two_sided([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], draws=2000, seed=0)
rho_b, p_b = s1.perm_p_two_sided([1, 2, 3, 4, 5, 6, 7, 8], [5, 1, 7, 3, 8, 2, 6, 4], draws=2000, seed=0)
check("C3 mediation detectable (rho=1, p<=0.05)", (round(rho_a, 6), p_a <= 0.05), (1.0, True))
check("C3 mediation absent (p>0.05)", p_b > 0.05, True)
# tie handling: t50 has ties by construction on a 10-step grid
check("C3 ties give defined rho", s1.spearman([330, 330, 350, 370], [.6, .61, .62, .63]) is not None, True)
# constant y must not raise, must return None
check("C3 constant arm -> rho None", s1.spearman([330, 340, 350], [.6, .6, .6]), None)

print("C4 defender cost")
check("C4 published constant is 15.70", round(2 * s1.Z_SUM ** 2, 2), 15.70)
check("C4 cost rises as sigma rises", s1.seed_budget(60, 10) > s1.seed_budget(38, 10), True)
check("C4 cost falls as delta rises", s1.seed_budget(38, 50) < s1.seed_budget(38, 10), True)

print("\nrehearsal demos:", sum(ok), "/", len(ok))
sys.exit(0 if all(ok) else 1)
