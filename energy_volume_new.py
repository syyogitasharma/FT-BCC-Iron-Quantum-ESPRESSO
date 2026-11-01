
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# BCC Fe: lattice parameters (Angstrom)
a = np.array([
    2.79998246,
    2.89999695,
    2.91999985,
    2.94000275,
    2.95000420,
    3.49997807,
    3.99999762
])

# DFT energies (eV/Fe)
# Enter only the matching energy values
# from the same DFT calculations.
E = np.array([
    -3459.67605955,
    -3459.77127164,
    -3459.78066107,
    -3459.77434163,
    -3459.77465524,
    -3458.67922131,
    -3457.42007932
])

# BCC primitive cell volume
V = a**3 / 2

# Sort by volume
order = np.argsort(V)
a, V, E = a[order], V[order], E[order]

# Use near-minimum points for fitting
mask = (a >= 2.79) & (a <= 3.01)
Vfit = V[mask]
Efit = E[mask]

if len(Vfit) < 4:
    raise ValueError(
        "Need at least 4 near-minimum points"
    )

# Birch-Murnaghan equation with B' fixed at 4
Bp = 4.0

def birch_murnaghan(V, E0, V0, B0):
    eta = (V0 / V)**(2/3)
    return E0 + (9 * V0 * B0 / 16) * (
        (eta - 1)**3 * Bp
        + (eta - 1)**2 * (6 - 4 * eta)
    )

i = np.argmin(Efit)

params, covariance = curve_fit(
    birch_murnaghan,
    Vfit,
    Efit,
    p0=[Efit[i], Vfit[i], 1.0],
    bounds=(
        [-np.inf, 0.1, 0.001],
        [np.inf, np.inf, 10]
    ),
    maxfev=20000
)

E0, V0, B0 = params

a0 = (2 * V0)**(1/3)
bulk_modulus = B0 * 160.21766208

print(f"Equilibrium lattice parameter: {a0:.5f} A")
print(f"Equilibrium primitive volume: {V0:.5f} A^3")
print(f"Bulk modulus: {bulk_modulus:.2f} GPa")
print(f"Pressure derivative B': {Bp:.3f} (fixed)")

# Generate fitted curve
Vcurve = np.linspace(min(Vfit), max(Vfit), 300)
Ecurve = birch_murnaghan(Vcurve, *params)

plt.figure(figsize=(8, 6))
plt.scatter(V, E, label="DFT data")
plt.plot(
    Vcurve, Ecurve,
    label="Birch-Murnaghan fit"
)
plt.xlabel("Volume per Fe atom (Angstrom^3)")
plt.ylabel("Energy (eV/Fe)")
plt.title("BCC Fe: Energy vs Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fe_energy_volume_new.png", dpi=300)
print("Graph saved: fe_energy_volume_new.png")
