# DIM (Dimensional Invariant Modulation) - Complete Documentation

**Dearman Cascade Wave: Resonant Harmonic Amplification and Vortex Dynamics in Swirling Flows**

**Author:** Shaun Robert DeArman  
**Date:** December 24, 2025

## Overview

This collection contains the complete documentation, theory, proofs, and implementation materials for the Dimensional Invariant Modulation (DIM) model - a novel extension to the Navier-Stokes equations incorporating resonant harmonic amplification based on 3-6-9 sequences and Larmor-like coupling.

## File Structure

### Core Documents
- **01_Dearman_Cascade_Wave_Abstract.txt** - Thesis abstract with keywords
- **02_Dearman_Cascade_Wave_Thesis_Manuscript.txt** - Complete thesis manuscript with all sections
- **03_Dearman_Cascade_Wave_Mathematical_Framework.txt** - Base equations and DIM extensions
- **04_Bibliography.txt** - Complete bibliography with citations
- **05_Supporting_Scripts.txt** - Python code examples for simulations
- **06_Complete_Project_Archive.txt** - Full project history, thought process, and choice tree
- **07_Complete_Mathematical_Formulation.txt** - Detailed mathematical formulation with LaTeX equations
- **08_Proofs_and_Citations.txt** - Mathematical proofs and expanded citations

### Additional Resources
- **10_Condensed_Thesis.txt** - Condensed thesis version for quick reference
- **11_Explanations.txt** - Base-level explanations (intro-friendly)
- **12_Implied_Magnetism_Model.py** - Complete Python implementation with classes and examples

## Theory Overview

### DIM Model Core Concepts

**Dimensional Invariant Modulation (DIM)** extends incompressible Navier-Stokes equations in cylindrical coordinates with:

1. **Resonance Amplification**: Selective boost to Fourier modes aligned with 3-6-9 patterns (freq mod 3 ≈ 0) and 124875-derived harmonics
2. **Larmor-like Coupling**: Weak gyroscopic term (ω_L × (r × ∇p)) for core stiffness
3. **Hopf Supplemental**: Added periodic forcing to Stuart-Landau equation → quasiperiodic tori, shifted bifurcation

### Key Predictions

- Delayed vortex breakdown (Re_crit ↑ dramatically by ~1000 units)
- Amplified circulation (~80-90%)
- Quasiperiodic ripples + discrete spectral peaks
- Stronger precession amplitude in 3D modes

### Applications

- Aerospace engineering (vortex devices)
- Energy systems
- Atmospheric modeling
- Materials science (implied magnetism)

## Mathematical Framework

The DIM model extends Navier-Stokes:

```
∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u + F_DIM

F_DIM = α * resonance_amplify(FFT(u)) + β * Larmor_cross(∇p, ω_L)
```

In Hopf normal form:
```
dA/dt = (μ + iω) A - l |A|^2 A + ∑ ε_k e^{i (k*3π t)}
```

## Results Summary

At Re=2000 (H/R=2.5):
- **Standard NS**: Double bubbles, Γ_max ≈1.1, variance=0
- **DIM**: No bubbles, Γ_max ≈1.95 (+80%), variance≈0.0005 (ripples)

Re sweep: DIM delays breakdown to >3000 (vs. standard ~1900)  
3D precession: DIM amplifies power 14x, adds 0.3/0.6 St peaks

Deviations: 8-15 standard deviations from literature means

## Experimental Validation

The framework provides falsifiable hypotheses:
- Delayed breakdown threshold (testable at Re=2500)
- Amplified circulation (measurable via velocity profiles)
- Spectral fingerprints (discrete peaks at 0.3k Strouhal)

## Related Concepts

### Implied Magnetism Theory

DIM also relates to the theory of universal magnetism - that all matter exhibits magnetic properties through electron spin and orbital motion. See explanations file for base-level introduction.

## Citation

If using this work, please cite:

DeArman, S. R. (2025). Dimensional Invariant Modulation (DIM): A Resonant Extension to Hopf Bifurcation in Swirling Flows. [Thesis/Preprint]

## License

All work is original, conducted on owned/licensed equipment by Shaun Robert DeArman.

## Contact

For questions or collaboration, contact the author.

