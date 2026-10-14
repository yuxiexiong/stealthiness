import unittest

import dir_common as dc


class Steps(unittest.TestCase):
    def test_saved_steps_cover_fork_and_end(self):
        s = dc.saved_steps(33, 48)
        self.assertEqual(s[0], 33)
        self.assertIn(68, s)
        self.assertTrue(all(b - a == 2 for a, b in zip(s, s[1:-1])) or all(b - a == 2 for a, b in zip(s, s[1:])))
        s2 = dc.saved_steps(57, 72)
        self.assertEqual((s2[0], s2[-1]), (57, 92))
        self.assertEqual(len(set(s2)), len(s2))


class Classes(unittest.TestCase):
    def base(self, **kw):
        m = {"cos_delta_tj_vs_d": 0.9, "restricted_tracks_mother": True, "excluded_stays_low": True, "progress_before_jump": True}
        m.update(kw)
        return m

    def test_direction_classes(self):
        self.assertEqual(dc.classify_direction(self.base()), "single_direction_hidden_progress")
        self.assertEqual(dc.classify_direction(self.base(cos_delta_tj_vs_d=0.5)), "direction_not_formed_at_onset")
        self.assertEqual(dc.classify_direction(self.base(restricted_tracks_mother=False)), "multi_directional")
        self.assertEqual(dc.classify_direction(self.base(excluded_stays_low=False)), "multi_directional")
        self.assertEqual(dc.classify_direction(self.base(progress_before_jump=False)), "no_hidden_progress")

    def test_intervention_classes(self):
        self.assertEqual(dc.classify_intervention({"suppress_t_jump": None, "amplify_t_jump": 60, "mother_t_jump": 72}), {"suppress": "blocked", "amplify": "advanced"})
        self.assertEqual(dc.classify_intervention({"suppress_t_jump": 80, "amplify_t_jump": 70, "mother_t_jump": 72}), {"suppress": "delayed", "amplify": "unchanged"})
        self.assertEqual(dc.classify_intervention({"suppress_t_jump": 73, "amplify_t_jump": None, "mother_t_jump": 72}), {"suppress": "unchanged", "amplify": "none"})

    def test_judge(self):
        d = {"s1": "single_direction_hidden_progress", "s2": "single_direction_hidden_progress", "s3": "single_direction_hidden_progress"}
        iv = {k: {"suppress": "blocked", "amplify": "advanced"} for k in d}
        v = dc.judge(d, iv)
        self.assertEqual((v["direction_tier"], v["necessity"], v["amplification"]), ("single_direction_hidden_progress", "necessary", "amplification_advances"))
        iv["s2"] = {"suppress": "delayed", "amplify": "unchanged"}
        v = dc.judge({**d, "s3": "multi_directional"}, iv)
        self.assertEqual((v["direction_tier"], v["necessity"], v["amplification"]), ("inconsistent", "partly", "amplification_advances"))


if __name__ == "__main__":
    unittest.main()
