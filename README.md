# TestSQRLab Environment

This repository contains Jupyter Notebooks and test scripts for running simulations using the custom SQRLab.jl package.

### 1. Setup & Linking the Development Package

Because SQRLab is a local package in development, you cannot simply Pkg.add("SQRLab"). You must link it via your local file system using development mode.

Activating the Environment

    1. Open your terminal in the TestSQRLab folder.

    2. Open the Julia REPL by typing julia.

    3. Press ] to enter the Package Manager (pkg>).

    4. Run activate . to activate the TestSQRLab environment.

Linking SQRLab

While still in the Package Manager (pkg>), run:

``` 
dev "C:/Users/WHERE_YOU_IMPORT_PACKAGE/IonSim_Package"  # Adjust path as necessary
```


### 2. The Revise.jl Workflow (Crucial)

If you modify source code inside the SQRLab repository, Jupyter Notebooks will not see the changes unless you restart the kernel. To fix this, always use Revise.jl.

At the top of your Jupyter Notebooks, run this block:

using Revise   # MUST be imported first!
using SQRLab
using PyPlot   # Or your preferred plotting library


Now, whenever you save a file in SQRLab (like Pulse.jl), your notebook will instantly use the updated code without requiring a kernel restart.

### 3. Running Simulations & Plotting

When plotting, pay close attention to the time steps. QuantumOptics and IonSim generally use standard SI units, but SQRLab scales times to microseconds (1e-6) internally for convenience.

Example: Running a fast Ramsey Sweep

Notice how we use Ref here to avoid a 10-hour compilation time!

```
using Revise
using SQRLab
using PyPlot

# 1. Setup Chamber
chamber = SQRLab.Trap.create_standard_chamber()

# 2. Define the sweep parameters
wait_times = 4.0:0.1:20.0  # Sweep from 4us to 20us
current_wait = Ref(wait_times[1])

# 3. Compile the Hamiltonian ONCE using the dynamic pointer
h = SQRLab.Pulse.pulse_dynamic(chamber, current_wait, 4e-6)

# 4. Run the loop
populations = Float64[]

for w in wait_times
    current_wait[] = w  # Update the pointer instantly
    
    # Run simulation. tspan must reach 'w'.
    tout, pop_e = SQRLab.Simulation.run_simulation_hamiltonian(chamber, h, 0:0.1:w)
    
    # Save the ground state population (1 - Excited)
    push!(populations, 1.0 - real(pop_e[end]))
end

# 5. Plotting
figure(figsize=(10, 6))
plot(wait_times, populations, label="Ground State", color="C0", linewidth=2)
xlabel("Wait Time (μs)")
ylabel("Population")
title("Ramsey Fringes")
ylim(0, 1.05)
grid(true, linestyle="--")
legend()
display(gcf())
```

### Plotting Rules to Remember:

1. Arrays vs Numbers: run_simulation_hamiltonian returns a full array of populations over time. If you are doing a sweep, only grab the last element (pop[end]).

2. Real Numbers: The solver returns complex floats (ComplexF64). Always use real(pop[end]) before plotting to satisfy the plotting libraries.

3. Time Vector matches Pulse: Ensure your run_simulation_hamiltonian time vector (0:0.1:w) reaches exactly to the end of your scheduled sequence, or your pulses will fall out of bounds.