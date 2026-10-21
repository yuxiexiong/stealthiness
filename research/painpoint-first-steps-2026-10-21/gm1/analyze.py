"""GM1 frozen analysis: is the poison set's whitened mean shift large enough to cross the BBP
spectral detectability threshold at realistic scale?

s      = || Sigma_clean^{-1/2} (mu_poison - mu_clean) ||   (Ledoit-Wolf shrinkage on clean rows)
k*(N)  = sqrt(D * N) / s^2   -- poisons needed so that (k/N) s^2 > sqrt(D/N)  (isotropic Gaussian BBP; 推导)
null   = s for 200 random clean pseudo-poison sets of the same size; "