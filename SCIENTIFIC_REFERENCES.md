# Scientific Methods and References

## The 4 Core Scientific Methods Implemented

This validation framework implements **4 peer-reviewed scientific methods** from computational biology and molecular dynamics literature:

---

## 1. Chapman-Kolmogorov Test for MSM Validation

**What it does:** Validates that the Markov State Model satisfies the Markov property by testing if predicted transition probabilities match estimated ones at multiple lag times.

**Scientific Paper:**
```
Prinz, J.-H., Wu, H., Sarich, M., Keller, B., Senne, M., Held, M., Chodera, J.D., 
Schütte, C., & Noé, F. (2011).
"Markov models of molecular kinetics: Generation and validation"
Journal of Chemical Physics, 134(17), 174105.
DOI: 10.1063/1.3565032
```

**Key Concept:** The Chapman-Kolmogorov equation states that P(k×τ) = P(τ)^k. If this holds, the model correctly captures the system's kinetics.

**Implementation:** `msm/validation.py` - function `chapman_kolmogorov_test()`

---

## 2. VAMP-2 Score for Model Selection

**What it does:** Provides a variational score to select optimal parameters (lag time and dimensionality) for time-lagged independent component analysis (tICA). Prevents overfitting by using validation sets.

**Scientific Paper:**
```
Wu, H., & Noé, F. (2020).
"Variational Approach for Learning Markov Processes from Time Series Data"
Journal of Nonlinear Science, 30, 23-66.
DOI: 10.1007/s00332-019-09567-y
```

**Key Concept:** VAMP-2 maximizes the sum of squared singular values of the Koopman operator approximation, identifying slow collective motions.

**Implementation:** `msm/validation.py` - function `vamp2_cross_validation()`

---

## 3. Time-lagged Independent Component Analysis (tICA)

**What it does:** Dimensionality reduction method that identifies the slowest collective motions in molecular dynamics, which are typically the most functionally relevant.

**Scientific Paper:**
```
Pérez-Hernández, G., Paul, F., Giorgino, T., De Fabritiis, G., & Noé, F. (2013).
"Identification of slow molecular order parameters for Markov model construction"
Journal of Chemical Physics, 139(1), 015102.
DOI: 10.1063/1.4811489
```

**Key Concept:** Unlike PCA (which finds maximum variance), tICA finds directions of slowest motion by maximizing autocorrelation at a lag time τ.

**Implementation:** Used throughout the pipeline; validation in `msm/validation.py`

---

## 4. Bootstrap Method for Uncertainty Quantification

**What it does:** Estimates confidence intervals for MSM parameters (stationary distributions, transition matrices, mean first passage times) by resampling the trajectory data.

**Scientific Paper:**
```
Trendelkamp-Schroer, B., Wu, H., Paul, F., & Noé, F. (2015).
"Estimation and uncertainty of reversible Markov models"
Journal of Chemical Physics, 143(17), 174101.
DOI: 10.1063/1.4934536
```

**Key Concept:** Bayesian bootstrap provides rigorous uncertainty estimates without parametric assumptions, essential for quantifying reliability of rare state populations.

**Implementation:** Existing in `msm/bootstrap_msm.py`; validated in `msm/validation.py`

---

## Additional Supporting Methods

### Implied Timescales Convergence Test
**Paper:** Same as #1 (Prinz et al. 2011)
**Purpose:** Checks that MSM timescales plateau with increasing lag time, confirming the Markov property.

### Stationary Distribution Validation
**Paper:** Same as #1 (Prinz et al. 2011)
**Purpose:** Compares MSM stationary distribution to empirical frequencies from the trajectory.

---

## Summary Table for Your Professors

| Method | Paper | Journal | Year | DOI |
|--------|-------|---------|------|-----|
| **Chapman-Kolmogorov Test** | Prinz et al. | J. Chem. Phys. | 2011 | 10.1063/1.3565032 |
| **VAMP-2 Scoring** | Wu & Noé | J. Nonlinear Sci. | 2020 | 10.1007/s00332-019-09567-y |
| **tICA** | Pérez-Hernández et al. | J. Chem. Phys. | 2013 | 10.1063/1.4811489 |
| **Bootstrap Uncertainty** | Trendelkamp-Schroer et al. | J. Chem. Phys. | 2015 | 10.1063/1.4934536 |

---

## Full Citations (APA Format)

1. **Prinz, J.-H., Wu, H., Sarich, M., Keller, B., Senne, M., Held, M., Chodera, J. D., Schütte, C., & Noé, F. (2011).** Markov models of molecular kinetics: Generation and validation. *Journal of Chemical Physics*, *134*(17), 174105. https://doi.org/10.1063/1.3565032

2. **Wu, H., & Noé, F. (2020).** Variational approach for learning Markov processes from time series data. *Journal of Nonlinear Science*, *30*, 23-66. https://doi.org/10.1007/s00332-019-09567-y

3. **Pérez-Hernández, G., Paul, F., Giorgino, T., De Fabritiis, G., & Noé, F. (2013).** Identification of slow molecular order parameters for Markov model construction. *Journal of Chemical Physics*, *139*(1), 015102. https://doi.org/10.1063/1.4811489

4. **Trendelkamp-Schroer, B., Wu, H., Paul, F., & Noé, F. (2015).** Estimation and uncertainty of reversible Markov models. *Journal of Chemical Physics*, *143*(17), 174101. https://doi.org/10.1063/1.4934536

---

## Additional Important References

### General MSM Theory
- **Noé, F., Schütte, C., Vanden-Eijnden, E., Reich, L., & Weikl, T. R. (2009).** Constructing the equilibrium ensemble of folding pathways from short off-equilibrium simulations. *Proceedings of the National Academy of Sciences*, *106*(45), 19011-19016. https://doi.org/10.1073/pnas.0905466106

### Software Implementation
- **Hoffmann, M., Scherer, M., Hempel, T., Mardt, A., de Silva, B., Husic, B. E., Klus, S., Wu, H., Kutz, N., Brunton, S. L., & Noé, F. (2021).** Deeptime: A Python library for machine learning dynamical models from time series data. *Machine Learning: Science and Technology*, *3*(1), 015009. https://doi.org/10.1088/2632-2153/ac3de0

---

## How to Cite This Work

If you use these validation tools in your research, please cite the four core papers above. Additionally, cite the software libraries:

**Deeptime (for tICA and MSM):**
```
Hoffmann, M., et al. (2021). Deeptime: A Python library for machine learning 
dynamical models from time series data. Machine Learning: Science and Technology, 
3(1), 015009.
```

**This Framework:**
```
[Your repository/paper citation when published]
Implements validation methods from Prinz et al. (2011), Wu & Noé (2020), 
Pérez-Hernández et al. (2013), and Trendelkamp-Schroer et al. (2015).
```

---

## Why These Methods Matter

### Scientific Rigor
All four methods are:
- ✅ **Peer-reviewed** in high-impact computational chemistry journals
- ✅ **Widely adopted** by the molecular dynamics community
- ✅ **Theoretically grounded** in statistical mechanics and dynamical systems theory
- ✅ **Validated** on numerous protein systems

### Practical Impact
Using these methods ensures:
1. **Markov property is satisfied** (Chapman-Kolmogorov) → Model captures correct kinetics
2. **Parameters are optimal** (VAMP-2) → No overfitting, best signal separation
3. **Slow motions identified** (tICA) → Focus on functionally relevant dynamics
4. **Uncertainties quantified** (Bootstrap) → Know reliability of predictions

---

## Questions Your Professors Might Ask

**Q: Why these specific papers?**
A: These are the foundational papers that introduced and validated each method. They have 1000+ citations combined and are considered standard references in computational biophysics.

**Q: Are these methods still current?**
A: Yes! The most recent paper (Wu & Noé 2020) is cutting-edge. tICA and MSM validation (2011-2015) are now standard practice in the field.

**Q: What journals are these?**
A: *Journal of Chemical Physics* (impact factor ~4) is the premier journal for computational molecular science. *Journal of Nonlinear Science* is a top applied mathematics journal.

**Q: Who are the key authors?**
A: **Frank Noé** (co-author on all 4 papers) is a leader in machine learning for molecular dynamics at Freie Universität Berlin. His group developed the deeptime/PyEMMA software packages used worldwide.

---

## Where to Find These Papers

All papers are available through:
1. **DOI links** (provided above) - use https://doi.org/[DOI]
2. **Google Scholar** - search by title
3. **University library access** - most universities have subscriptions
4. **arXiv preprints** - some authors post preprints

---

*This document provides complete scientific backing for the validation framework. All methods are rigorously tested and widely used in computational biology research.*
