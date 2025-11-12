import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# ========================
# INPUT SECTION
# ========================
# Input your volume fraction (φ) and conductivity (σ) data directly.
# Make sure the number of items in both arrays is the same.

# Example (replace these with your own data)
# Volume fraction (φ) of the conductive filler
# These values should be fractions (e.g., 0.10 for 10%)
phi_filler = np.array([0, 0.42342, 0.594934, 0.746029])  # example

# Corresponding conductivity values (S/m)
# Make sure these match the order of your phi_filler values
sigma_exp = np.array([23958.9822,22020.12,27531.144,30943.466]) # example percolation data

# ===================================================================
# --- No more conversions needed ---
# The old "MATERIAL DENSITIES" and "Wt% vs. Vol. Frac. Table"
# sections have been removed. The program now uses your
# 'phi_filler' array directly.
# ===================================================================


# ========================
# STEP 3 & 4: Grid search for φ_c (Route A)
# ========================
phi_c_candidates = np.arange(min(phi_filler) + 0.01, max(phi_filler) - 0.01, 0.005)

best_R2 = -np.inf
best_fit = {}

# Handle cases where no fit is found
if len(phi_c_candidates) == 0:
    print("Error: Could not find valid phi_c candidates.")
    print("This often happens if you have too few data points or")
    print("if the min/max volume fractions are too close.")
    # Set a dummy best_fit to avoid crashing later
    best_fit = {'phi_c': 0, 't': 0, 'a': 0, 'R2': 0, 'sigma0': 0}
else:
    for phi_c in phi_c_candidates:
        mask = phi_filler > phi_c + 0.002
        if mask.sum() < 3:
            continue

        x = np.log(phi_filler[mask] - phi_c)
        y = np.log(sigma_exp[mask])

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        if r_value**2 > best_R2:
            best_R2 = r_value**2
            best_fit = {
                'phi_c': phi_c,
                't': slope,
                'a': intercept,
                'R2': r_value**2,
                'sigma0': np.exp(intercept)
            }

if best_R2 == -np.inf:
     print("Error: A fit could not be found. Check your data.")
     print("A minimum of 3 data points above phi_c is required.")
     # Set a dummy best_fit to avoid crashing
     best_fit = {'phi_c': 0, 't': 0, 'a': 0, 'R2': 0, 'sigma0': 0}

# ========================================
# STEP 5: Diagnostics, Save Data, & Plots
# ========================================
phi_c = best_fit['phi_c']
t = best_fit['t']
sigma0 = best_fit['sigma0']

# Create results folder
os.makedirs('percolation_fit_results', exist_ok=True)

print("\nBest fit results:")
for k, v in best_fit.items():
    print(f"{k:8s} : {v}")

# --- Define data for Plot 1: Log–linear fit ---
mask_best = phi_filler > phi_c + 0.002
# Scatter data (the points used for the best fit)
x_scatter1 = np.log(phi_filler[mask_best] - phi_c)
y_scatter1 = np.log(sigma_exp[mask_best])
# Line data (the prediction)
phi_fit = phi_filler[phi_filler > phi_c]
sigma_pred = sigma0 * (phi_fit - phi_c) ** t
x_line1 = np.log(phi_fit - phi_c)
y_line1 = np.log(sigma_pred)

# --- Define data for Plot 2: σ vs φ ---
# Scatter data (all experimental points)
x_scatter2 = phi_filler
y_scatter2 = sigma_exp
# Line data (the prediction, same as plot 1)
x_line2 = phi_fit
y_line2 = sigma_pred

# --- Define data for Plot 3: Residuals ---
residuals = y_scatter1 - (best_fit['a'] + best_fit['t'] * x_scatter1)
# Scatter data
x_scatter3 = y_scatter1 # This is ln(sigma_exp[mask])
y_scatter3 = residuals

# ========================================
# NEW: SAVE PLOT DATA TO TEXT FILES
# ========================================
file_header = "Data for Origin plotting. Column 1: X, Column 2: Y"

# Save Plot 1 data
data_plot1_scatter = np.vstack((x_scatter1, y_scatter1)).T
np.savetxt('percolation_fit_results/plot1_loglinear_scatter.txt', data_plot1_scatter, header=f"{file_header}\nx: ln(phi - phi_c) \ny: ln(sigma_exp)", delimiter='\t')
data_plot1_line = np.vstack((x_line1, y_line1)).T
np.savetxt('percolation_fit_results/plot1_loglinear_fit_line.txt', data_plot1_line, header=f"{file_header}\nx: ln(phi_fit - phi_c) \ny: ln(sigma_pred)", delimiter='\t')

# Save Plot 2 data
data_plot2_scatter = np.vstack((x_scatter2, y_scatter2)).T
np.savetxt('percolation_fit_results/plot2_sigma_vs_phi_scatter.txt', data_plot2_scatter, header=f"{file_header}\nx: phi_filler \ny: sigma_exp", delimiter='\t')
data_plot2_line = np.vstack((x_line2, y_line2)).T
np.savetxt('percolation_fit_results/plot2_sigma_vs_phi_fit_line.txt', data_plot2_line, header=f"{file_header}\nx: phi_fit \ny: sigma_pred", delimiter='\t')

# Save Plot 3 data
data_plot3_scatter = np.vstack((x_scatter3, y_scatter3)).T
np.savetxt('percolation_fit_results/plot3_residuals_scatter.txt', data_plot3_scatter, header=f"{file_header}\nx: ln(sigma_exp) \ny: Residuals", delimiter='\t')

print("\nData for all plots saved to .txt files in 'percolation_fit_results/'")

# ========================
# PLOTTING SECTION
# ========================

# Plot 1: Log–linear fit
plt.figure()
plt.scatter(x_scatter1, y_scatter1, label='Data')
plt.plot(x_line1, y_line1, label='Fit', lw=2)
plt.xlabel('ln(φ - φ_c)')
plt.ylabel('ln(σ)')
plt.legend()
plt.title('Log–Linear Fit (Route A)')
plt.savefig('percolation_fit_results/loglinear_fit.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: σ vs φ
plt.figure()
plt.scatter(x_scatter2, y_scatter2, label='Experimental')
plt.plot(x_line2, y_line2, label='Fit', lw=2)
plt.yscale('log')
plt.xlabel('Volume fraction φ (filler)')
plt.ylabel('Conductivity σ (S/m)')
plt.legend()
plt.title('Conductivity vs Filler Volume Fraction')
plt.savefig('percolation_fit_results/sigma_vs_phi.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Residuals
plt.figure()
plt.scatter(x_scatter3, y_scatter3)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('ln(σ)')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('percolation_fit_results/residuals_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll plots saved to 'percolation_fit_results/' directory.")
