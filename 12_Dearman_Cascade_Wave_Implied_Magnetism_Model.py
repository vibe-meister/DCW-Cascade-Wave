"""
DIM (Dimensional Invariant Modulation) - Implied Magnetism Model
Author: Shaun Robert DeArman
Date: December 24, 2025

This module implements the DIM model for fluid dynamics simulations
with resonance amplification and Larmor-like coupling.

For complete documentation, see README.md and related documentation files.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

class DIMFluidSimulator:
    """
    Core DIM fluid dynamics simulator for axisymmetric swirling flows.
    
    Implements:
    - Navier-Stokes with DIM extensions
    - Resonance amplification (3-6-9 patterns)
    - Larmor-like coupling
    - Streamfunction-circulation formulation
    """
    
    def __init__(self, Re=2000, nr=50, nz=100, H=2.5, dim_on=True):
        """
        Initialize DIM simulator.
        
        Parameters:
        -----------
        Re : float
            Reynolds number
        nr, nz : int
            Grid resolution in radial and axial directions
        H : float
            Aspect ratio (height/radius)
        dim_on : bool
            Enable DIM extensions
        """
        self.Re = Re
        self.nr = nr
        self.nz = nz
        self.H = H
        self.dim_on = dim_on
        
        # Grid setup
        self.dr = 1.0 / (nr - 1)
        self.dz = H / (nz - 1)
        self.r = np.linspace(0, 1, nr)
        self.z = np.linspace(0, H, nz)
        R, Z = np.meshgrid(self.r, self.z, indexing='ij')
        self.R = R
        self.Z = Z
        
        # Flow fields
        self.psi = np.zeros((nr, nz))  # Streamfunction
        self.gamma = np.zeros((nr, nz))  # Circulation
        
        # DIM parameters
        self.res_freqs = [3, 6, 9, 12, 18, 24, 48, 75]  # 3-6-9 harmonics
        self.larmor_omega = 0.1
        self.alpha = 0.05  # Resonance strength
        self.beta = 0.1    # Larmor strength
        
        # Time tracking
        self.t = 0.0
        self.dt = 0.01
        
    def resonance_term(self, field):
        """
        Compute DIM resonance amplification term.
        
        Amplifies modes where frequency mod 3 ≈ 0, scaled by 124875 sequences.
        """
        if not self.dim_on:
            return np.zeros_like(field)
        
        # Simplified resonance amplification
        # In practice, this would use FFT and selective mode boosting
        res_contribution = np.zeros_like(field)
        
        for freq in self.res_freqs:
            # Harmonic injection at resonance frequencies
            res_contribution += self.alpha * np.sin(2 * np.pi * freq * self.t) * field
        
        return res_contribution
    
    def larmor_term(self, pressure_grad_r, pressure_grad_z):
        """
        Compute Larmor-like coupling term.
        
        L = ω_L (r × ∇p) · ê_θ / r
        """
        if not self.dim_on:
            return np.zeros_like(self.gamma)
        
        # Larmor coupling: β ω_L (r ∂p/∂z - z ∂p/∂r) / r
        larmor = self.beta * self.larmor_omega * (
            self.R * pressure_grad_z - self.Z * pressure_grad_r
        ) / (self.R + 1e-10)  # Avoid division by zero
        
        return larmor
    
    def compute_pressure_gradient(self):
        """
        Compute pressure gradient from velocity field.
        Simplified approximation for demonstration.
        """
        # In full implementation, solve Poisson equation for pressure
        # Here: simplified gradient estimate
        p_grad_r = -np.gradient(self.gamma, axis=0) / self.dr
        p_grad_z = -np.gradient(self.gamma, axis=1) / self.dz
        
        return p_grad_r, p_grad_z
    
    def step(self):
        """
        Advance simulation one time step using RK4.
        """
        # Compute pressure gradients
        p_grad_r, p_grad_z = self.compute_pressure_gradient()
        
        # Advection term (simplified)
        adv_r = np.gradient(self.gamma, axis=0) / self.dr
        adv_z = np.gradient(self.gamma, axis=1) / self.dz
        advection = adv_r + adv_z
        
        # Diffusion term
        laplacian_r = np.gradient(np.gradient(self.gamma, axis=0), axis=0) / (self.dr**2)
        laplacian_z = np.gradient(np.gradient(self.gamma, axis=1), axis=1) / (self.dz**2)
        diffusion = (1.0 / self.Re) * (laplacian_r + laplacian_z - self.gamma / (self.R**2 + 1e-10))
        
        # DIM terms
        resonance = self.resonance_term(self.gamma)
        larmor = self.larmor_term(p_grad_r, p_grad_z)
        
        # Update
        dgamma_dt = -advection + diffusion + resonance + larmor
        self.gamma += self.dt * dgamma_dt
        
        self.t += self.dt
    
    def run(self, t_max=50):
        """
        Run simulation to specified time.
        
        Returns:
        --------
        gamma_max : float
            Maximum circulation
        variance : float
            Temporal variance (quasiperiodicity measure)
        """
        n_steps = int(t_max / self.dt)
        
        # Store time series for variance calculation
        gamma_history = []
        
        for i in range(n_steps):
            self.step()
            if i % 100 == 0:  # Sample every 100 steps
                gamma_history.append(self.gamma.copy())
        
        gamma_max = np.max(np.abs(self.gamma))
        
        # Compute variance from history
        if len(gamma_history) > 1:
            variance = np.var([np.mean(g) for g in gamma_history])
        else:
            variance = 0.0
        
        return gamma_max, variance
    
    def has_breakdown(self):
        """
        Detect vortex breakdown (recirculation bubbles).
        
        Simplified: check for negative axial velocity regions.
        """
        # In full implementation, analyze streamfunction topology
        # Here: simplified check for circulation reversal
        return np.any(self.gamma < -0.1)


class DIMHopfAmplitude:
    """
    DIM-extended Hopf bifurcation amplitude equations.
    
    Implements resonant forced Stuart-Landau equation for 3D precession modes.
    """
    
    def __init__(self, mu=0.1, omega=0.04, l=1.0, dim_on=True):
        """
        Initialize Hopf amplitude equation solver.
        
        Parameters:
        -----------
        mu : float
            Growth rate parameter (Re - Re_crit)
        omega : float
            Natural frequency
        l : float
            Landau coefficient
        dim_on : bool
            Enable DIM resonant forcing
        """
        self.mu = mu
        self.omega = omega
        self.l = l
        self.dim_on = dim_on
        
        # DIM resonance frequencies (3-6-9 multiples)
        self.res_freqs = [0.3, 0.6, 0.9]
        self.epsilon_k = 0.05  # Forcing amplitude
        self.beta = 0.1
        self.omega_L = 0.1
    
    def hopf_dim(self, y, t):
        """
        DIM-extended Hopf normal form.
        
        dA/dt = (μ + i ω_0) A - ℓ |A|² A + F_res(t) + i Δω_L A
        """
        A = y[0] + 1j * y[1]
        
        # Standard Hopf term
        dA = (self.mu + 1j * self.omega) * A - self.l * abs(A)**2 * A
        
        # DIM resonant forcing
        if self.dim_on:
            for k, f_k in enumerate(self.res_freqs):
                dA += self.epsilon_k * np.exp(1j * 2 * np.pi * f_k * t)
            
            # Larmor frequency shift
            dA += 1j * self.beta * self.omega_L * A
        
        return [dA.real, dA.imag]
    
    def solve(self, t_span, y0=[0.1, 0.0]):
        """
        Solve amplitude equation.
        
        Returns:
        --------
        t : array
            Time points
        y : array
            Solution [Re(A), Im(A)]
        """
        t = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) * 100))
        y = odeint(self.hopf_dim, y0, t)
        return t, y
    
    def compute_psd(self, t, y):
        """
        Compute power spectral density of |A(t)|.
        """
        A = y[:, 0] + 1j * y[:, 1]
        A_mag = np.abs(A)
        
        # Welch PSD
        from scipy.signal import welch
        f, psd = welch(A_mag, 1.0 / (t[1] - t[0]), nperseg=len(t) // 4)
        
        return f, psd


def run_comparison_sweep(Re_list, dim_on=True):
    """
    Run Reynolds number sweep comparison.
    
    Parameters:
    -----------
    Re_list : array
        Reynolds numbers to test
    dim_on : bool
        Enable DIM extensions
    
    Returns:
    --------
    results : list
        [(Re, breakdown, gamma_max), ...]
    """
    results = []
    
    for Re in Re_list:
        sim = DIMFluidSimulator(Re=Re, dim_on=dim_on)
        gamma_max, variance = sim.run(t_max=50)
        breakdown = sim.has_breakdown()
        results.append((Re, breakdown, gamma_max, variance))
        print(f"Re={Re}: breakdown={breakdown}, Γ_max={gamma_max:.3f}, var={variance:.6f}")
    
    return results


# Example usage
if __name__ == "__main__":
    print("DIM Fluid Dynamics Simulator")
    print("=" * 40)
    
    # Test at Re=2000
    print("\n1. Standard NS (DIM off):")
    sim_std = DIMFluidSimulator(Re=2000, dim_on=False)
    gamma_max_std, var_std = sim_std.run(t_max=50)
    breakdown_std = sim_std.has_breakdown()
    print(f"   Breakdown: {breakdown_std}")
    print(f"   Γ_max: {gamma_max_std:.3f}")
    print(f"   Variance: {var_std:.6f}")
    
    print("\n2. DIM Model (DIM on):")
    sim_dim = DIMFluidSimulator(Re=2000, dim_on=True)
    gamma_max_dim, var_dim = sim_dim.run(t_max=50)
    breakdown_dim = sim_dim.has_breakdown()
    print(f"   Breakdown: {breakdown_dim}")
    print(f"   Γ_max: {gamma_max_dim:.3f}")
    print(f"   Variance: {var_dim:.6f}")
    
    print(f"\n3. Amplification: {(gamma_max_dim/gamma_max_std - 1)*100:.1f}%")
    
    # Re sweep
    print("\n4. Reynolds number sweep:")
    Re_list = np.arange(1500, 3100, 200)
    results = run_comparison_sweep(Re_list, dim_on=True)
    
    # 3D precession analysis
    print("\n5. 3D Precession (Hopf amplitude):")
    hopf_dim = DIMHopfAmplitude(mu=0.1, omega=0.04, dim_on=True)
    t, y = hopf_dim.solve([0, 2000])
    f, psd = hopf_dim.compute_psd(t, y)
    
    print(f"   Peak frequency: {f[np.argmax(psd)]:.4f} St")
    print(f"   Peak power: {np.max(psd):.2e}")

