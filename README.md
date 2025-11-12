This repository contains a Python script to model and analyze conductivity in composite materials using standard percolation theory.

## Overview
This code implements a standard percolation model, which is used to describe systems where a conductive filler is added to an **insulating host**.

The model fits experimental data to the classic percolation power-law equation:

$$
\sigma = S(\phi - \phi_c)^t
$$

...where:
* $\sigma$ is the bulk conductivity of the composite
* $\phi$ is the volume fraction of the conductive filler
* $\phi_c$ is the critical percolation threshold (the point where a conductive path first forms)
* $S$ and $t$ are critical exponents related to the material and lattice structure

The script uses a grid-search algorithm to find the best-fit parameters ($\phi_c$, $S$, and $t$) that minimize the error for a given experimental dataset.



## How to Use
1.  **Open the Script:** Open the `.py` file in any code editor.
2.  **Update Data:** Locate the data lists (e.g., `phi_data` and `sigma_data`) at the top of the script.
3.  **Paste Your Data:** Replace the placeholder values with your own experimental data.
4.  **Run Script:** Execute the Python script.
5.  **Get Results:** The script will print the best-fit parameters ($\phi_c$, $S$, and $t$) to the console.
6.  **Plotting:** The code also generates a corresponding `.txt` file with the fitted curve, which can be easily imported into software like Origin for plotting against your experimental data.

## Key Features
* Fits experimental conductivity data to the standard percolation model.
* Calculates the critical percolation threshold ($\phi_c$) and exponents ($S$, $t$).
* Uses a grid search to find the optimal parameters.
* Outputs a simple `.txt` file ready for visualization in scientific plotting software.
