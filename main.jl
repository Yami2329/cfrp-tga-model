# =============================================================================
# CFRP Oxidative Decomposition Modeling
# Homework 4 – Introduction to Scientific Computing
#
# Tasks:
#   A: Forward Model
#   B: Synthetic Data Generation
#   C: Inverse Problem Setup
#   D: Parameter Estimation & Validation
# =============================================================================

using Catalyst 
using ModelingToolkit
using OrdinaryDiffEq
using DataFrames, Random
using PEtab
using OptimizationOptimJL
using Optim
using Plots
using Printf
using Statistics

# =========================
# SET OUTPUT DIRECTORY
# =========================
mkpath("C:/Users/user/results")
cd("C:/Users/user/results")

# =============================================================================
# Task A: Forward Model
# =============================================================================

using Catalyst, ModelingToolkit

"""
    build_cfrp_reaction_system()

Build the Catalyst.jl ReactionSystem for CFRP oxidative decomposition.

Species: M (Matrix), C (Char), F (Fiber), Temp (Temperature), G1 (Volatiles), G2 (Oxidation gas)
Parameters: log10_A1, E1, n1, nu_char, log10_A2, E2, n2, m2, log10_A3, E3, n3, m3, PO2, beta

Returns a completed ReactionSystem ready for simulation.
"""
function build_cfrp_reaction_system()
    # Independent variable
    t = default_t()

    # Define species
    @species M(t) C(t) F(t) Temp(t) G1(t) G2(t)

    # Define parameters
    # Kinetic parameters (log10 of pre-exponential for numerical stability)
    @parameters log10_A1 E1 n1 nu_char
    @parameters log10_A2 E2 n2 m2
    @parameters log10_A3 E3 n3 m3
    # Experimental condition parameters
    @parameters PO2 beta

    # Gas constant [J/(mol·K)]
    R_gas = 8.314

    # --- Rate expressions ---
    # Use max(species, 0) to prevent NaN from negative^fractional during optimization
    # Reaction 1: Matrix Pyrolysis (anaerobic)
    # r1 = A1 * exp(-E1/(R*T)) * M^n1
    r1 = (10.0^log10_A1) * exp(-E1 / (R_gas * Temp)) * max(M, 0.0)^n1

    # Reaction 2: Char Oxidation (aerobic)
    # r2 = A2 * exp(-E2/(R*T)) * C^n2 * PO2^m2
    r2 = (10.0^log10_A2) * exp(-E2 / (R_gas * Temp)) * max(C, 0.0)^n2 * PO2^m2

    # Reaction 3: Fiber Oxidation (aerobic)
    # r3 = A3 * exp(-E3/(R*T)) * F^n3 * PO2^m3
    r3 = (10.0^log10_A3) * exp(-E3 / (R_gas * Temp)) * max(F, 0.0)^n3 * PO2^m3

    # --- Reactions ---
    rxns = [
        # Temperature ramp: dT/dt = beta  (linear heating program)
        Reaction(beta, nothing, [Temp], nothing, [1]),

        # Reaction 1: M → nu_char*C + (1-nu_char)*G1
        # only_use_rate=true: the rate expression IS the full propensity (not mass-action)
        Reaction(r1, [M], [C, G1], [1], [nu_char, 1 - nu_char]; only_use_rate=true),

        # Reaction 2: C → G2  (with O₂ dependence in rate)
        Reaction(r2, [C], [G2], [1], [1]; only_use_rate=true),

        # Reaction 3: F → G2  (with O₂ dependence in rate)
        Reaction(r3, [F], [G2], [1], [1]; only_use_rate=true),
    ]

    # Build and complete the reaction system
    @named cfrp = ReactionSystem(rxns, t)
    cfrp = complete(cfrp)

    return cfrp
end

# =============================================================================
# True parameters for data generation (Section 6 of assignment)
# =============================================================================
const TRUE_PARAMS = Dict(
    :log10_A1 => 5.0,       # log10(A1) for matrix pyrolysis
    :E1       => 120_000.0,  # Activation energy [J/mol]
    :n1       => 1.0,        # Reaction order for M
    :nu_char  => 0.2,        # Char yield (FIXED, not estimated)
    :log10_A2 => 7.0,        # log10(A2) for char oxidation
    :E2       => 160_000.0,  # Activation energy [J/mol]
    :n2       => 1.5,        # Reaction order for C
    :m2       => 0.6,        # Oxygen reaction order
    :log10_A3 => 9.0,        # log10(A3) for fiber oxidation
    :E3       => 250_000.0,  # Activation energy [J/mol]
    :n3       => 2.0,        # Reaction order for F
    :m3       => 0.85,       # Oxygen reaction order
)

# =============================================================================
# Initial conditions
# =============================================================================
const INITIAL_M    = 0.3    # Matrix mass fraction
const INITIAL_F    = 0.7    # Fiber mass fraction
const INITIAL_C    = 0.0    # Char mass fraction (none initially)
const INITIAL_TEMP = 300.0  # Starting temperature [K]
const INITIAL_G1   = 0.0    # No gas initially
const INITIAL_G2   = 0.0    # No gas initially

# =============================================================================
# Experimental conditions (Section 3)
# Note: β converted from K/min to K/s (kinetics use seconds!)
# =============================================================================
const EXPERIMENTS = Dict(
    "Exp1" => (beta = 2.5 / 60.0,  PO2 = 0.21),  # 2.5 K/min, 21% O₂
    "Exp2" => (beta = 5.0 / 60.0,  PO2 = 0.21),  # 5.0 K/min, 21% O₂
    "Exp3" => (beta = 10.0 / 60.0, PO2 = 0.21),  # 10.0 K/min, 21% O₂
    "Exp4" => (beta = 5.0 / 60.0,  PO2 = 0.05),  # 5.0 K/min, 5% O₂
)

# Final temperature for TGA simulation [K]
const T_FINAL = 1200.0


# =============================================================================
# Task B: Synthetic Data Generation
# =============================================================================


using OrdinaryDiffEq, DataFrames, Random

"""
    simulate_experiment(rn, exp_id, exp_cond, params; n_points=150)

Simulate a single TGA experiment using the Catalyst.jl reaction system.

Arguments:
- `rn`: The completed ReactionSystem
- `exp_id`: Experiment identifier string (e.g., "Exp1")
- `exp_cond`: Named tuple with (beta, PO2) for this experiment
- `params`: Dict of true parameter values
- `n_points`: Number of data points to sample

Returns: (times, temperatures, mass_values) arrays
"""
function simulate_experiment(rn, exp_id, exp_cond, params; n_points=150)
    # Compute simulation time span from temperature range
    # T = T0 + β*t  →  t_final = (T_final - T0) / β
    t_final = (T_FINAL - INITIAL_TEMP) / exp_cond.beta

    # Map species to initial conditions
    @unpack M, C, F, Temp, G1, G2 = rn
    u0 = [
        M    => INITIAL_M,
        C    => INITIAL_C,
        F    => INITIAL_F,
        Temp => INITIAL_TEMP,
        G1   => INITIAL_G1,
        G2   => INITIAL_G2,
    ]

    # Map all parameters (true values + experimental conditions)
    @unpack log10_A1, E1, n1, nu_char, log10_A2, E2, n2, m2,
            log10_A3, E3, n3, m3, PO2, beta = rn
    pmap = [
        log10_A1 => params[:log10_A1],
        E1       => params[:E1],
        n1       => params[:n1],
        nu_char  => params[:nu_char],
        log10_A2 => params[:log10_A2],
        E2       => params[:E2],
        n2       => params[:n2],
        m2       => params[:m2],
        log10_A3 => params[:log10_A3],
        E3       => params[:E3],
        n3       => params[:n3],
        m3       => params[:m3],
        PO2      => exp_cond.PO2,
        beta     => exp_cond.beta,
    ]

    # Create and solve the ODE problem
    oprob = ODEProblem(rn, u0, (0.0, t_final), pmap)
    sol = solve(oprob, Rodas5P(); abstol=1e-10, reltol=1e-8, saveat=t_final/n_points)

    # Extract results
    times = sol.t
    # Mass = M + C + F (solid mass fraction — excludes gases)
    mass_values = sol[M] .+ sol[C] .+ sol[F]
    temperatures = sol[Temp]

    return times, temperatures, mass_values
end

"""
    generate_all_synthetic_data(rn; noise_sigma=0.005, n_points=150, seed=42)

Generate noisy synthetic TGA data for all 4 experiments.

Arguments:
- `rn`: The completed ReactionSystem
- `noise_sigma`: Gaussian noise standard deviation (0.5% = 0.005)
- `n_points`: Number of data points per experiment
- `seed`: Random seed for reproducibility

Returns: (measurements_df, clean_data)
- `measurements_df`: PEtab-compatible DataFrame
- `clean_data`: Dict of clean simulation results for each experiment
"""
function generate_all_synthetic_data(rn; noise_sigma=0.005, n_points=150, seed=42)
    Random.seed!(seed)

    # Storage for clean data (for plotting reference)
    clean_data = Dict{String, NamedTuple}()

    # PEtab measurement DataFrame
    all_obs_ids    = String[]
    all_sim_ids    = String[]
    all_times      = Float64[]
    all_measurements = Float64[]

    for (exp_id, exp_cond) in sort(collect(EXPERIMENTS))
        println("  Simulating $exp_id: β = $(exp_cond.beta*60) K/min, PO₂ = $(exp_cond.PO2)")

        times, temperatures, mass_clean = simulate_experiment(
            rn, exp_id, exp_cond, TRUE_PARAMS; n_points=n_points
        )

        # Add Gaussian noise (σ = 0.5% of initial mass = 0.005)
        noise = noise_sigma .* randn(length(mass_clean))
        mass_noisy = mass_clean .+ noise
        # Clamp to physically reasonable range [0, 1]
        mass_noisy = clamp.(mass_noisy, 0.0, 1.0)

        # Store clean data for later plotting
        clean_data[exp_id] = (
            times = times,
            temperatures = temperatures,
            mass_clean = mass_clean,
            mass_noisy = mass_noisy,
        )

        # Append to PEtab measurement arrays
        for i in eachindex(times)
            push!(all_obs_ids, "mass_total")
            push!(all_sim_ids, exp_id)
            push!(all_times, times[i])
            push!(all_measurements, mass_noisy[i])
        end
    end

    # Build PEtab-compatible measurements DataFrame
    measurements_df = DataFrame(
        obs_id        = all_obs_ids,
        simulation_id = all_sim_ids,
        time          = all_times,
        measurement   = all_measurements,
    )

    return measurements_df, clean_data
end


# =============================================================================
# Tasks C & D: Inverse Problem Setup and Parameter Estimation
# =============================================================================
#
# Set up the PEtab model for multi-condition parameter estimation.
# Define simulation conditions, observables, and parameters to estimate.
# Run multi-start optimization and compare recovered vs true parameters.
# =============================================================================

using PEtab, OrdinaryDiffEq, OptimizationOptimJL, Optim, DataFrames, Printf, Random

"""
    setup_petab_model(rn, measurements_df)

Set up the PEtab model for parameter estimation.

Arguments:
- `rn`: The completed Catalyst.jl ReactionSystem
- `measurements_df`: PEtab-compatible measurement DataFrame

Returns: PEtabODEProblem ready for optimization
"""
function setup_petab_model(rn, measurements_df)
    @unpack M, C, F, Temp, G1, G2 = rn

    # --- Observables ---
    # We observe total solid mass = M + C + F
    # Noise model: constant Gaussian noise σ = 0.005 (0.5%)
    @parameters sigma_mass
    observables = [
        PEtabObservable(:mass_total, M + C + F, sigma_mass),
    ]

    # --- Simulation Conditions ---
    # Map each experiment ID to its specific β and PO₂ values
    # These override the model parameters for each experimental condition
    @unpack PO2, beta = rn
    simulation_conditions = [
        PEtabCondition(:Exp1, beta => 2.5 / 60.0,  PO2 => 0.21),
        PEtabCondition(:Exp2, beta => 5.0 / 60.0,  PO2 => 0.21),
        PEtabCondition(:Exp3, beta => 10.0 / 60.0, PO2 => 0.21),
        PEtabCondition(:Exp4, beta => 5.0 / 60.0,  PO2 => 0.05),
    ]

    # --- Parameters to Estimate ---
    # Hint from assignment: estimate log10(A) instead of A for numerical stability
    # We estimate 11 kinetic parameters + 1 noise parameter
    petab_parameters = [
        # Matrix Pyrolysis parameters
        PEtabParameter(:log10_A1, value=5.0,  lb=2.0,    ub=10.0,    scale=:lin),
        PEtabParameter(:E1,       value=1.2e5, lb=8e4,    ub=2e5,     scale=:lin),
        PEtabParameter(:n1,       value=1.0,  lb=0.5,    ub=3.0,     scale=:lin),
        # Char Oxidation parameters
        PEtabParameter(:log10_A2, value=7.0,  lb=3.0,    ub=12.0,    scale=:lin),
        PEtabParameter(:E2,       value=1.6e5, lb=1e5,    ub=2.5e5,   scale=:lin),
        PEtabParameter(:n2,       value=1.5,  lb=0.5,    ub=3.0,     scale=:lin),
        PEtabParameter(:m2,       value=0.6,  lb=0.1,    ub=2.0,     scale=:lin),
        # Fiber Oxidation parameters
        PEtabParameter(:log10_A3, value=9.0,  lb=5.0,    ub=13.0,    scale=:lin),
        PEtabParameter(:E3,       value=2.5e5, lb=1.5e5,  ub=3.5e5,   scale=:lin),
        PEtabParameter(:n3,       value=2.0,  lb=0.5,    ub=3.5,     scale=:lin),
        PEtabParameter(:m3,       value=0.85, lb=0.1,    ub=2.0,     scale=:lin),
        # Noise parameter (estimated alongside kinetics)
        PEtabParameter(:sigma_mass, value=0.005, lb=1e-4, ub=0.1,    scale=:log10),
    ]

    # --- Initial Conditions (same for all experiments) ---
    state_map = [
        M    => INITIAL_M,
        C    => INITIAL_C,
        F    => INITIAL_F,
        Temp => INITIAL_TEMP,
        G1   => INITIAL_G1,
        G2   => INITIAL_G2,
    ]

    # --- Fixed Parameters ---
    @unpack nu_char = rn
    parameter_map = [
        nu_char => 0.2,  # Char yield is FIXED (not estimated)
    ]

    # --- Build PEtab Model ---
    model = PEtabModel(
        rn,
        observables,
        measurements_df,
        petab_parameters;
        simulation_conditions = simulation_conditions,
        speciemap = state_map,
        parametermap = parameter_map,
        verbose = true,
    )

    # --- Build PEtab ODE Problem ---
    petab_prob = PEtabODEProblem(
        model;
        odesolver = ODESolver(Rodas5P(); abstol=1e-8, reltol=1e-6, maxiters=Int64(1e5), force_dtmin=true),
        gradient_method = :ForwardDiff,
        hessian_method = :ForwardDiff,
    )

    return petab_prob
end

"""
    run_parameter_estimation(petab_prob; n_multistarts=10)

Run multi-start optimization to estimate kinetic parameters.
Uses manual perturbations around nominal values to avoid the extreme
instabilities that random Latin Hypercube sampling causes in Arrhenius
parameter spaces (where 10^A * exp(-E/RT) easily overflows).

Arguments:
- `petab_prob`: PEtabODEProblem
- `n_multistarts`: Number of starting points (1 nominal + perturbations)

Returns: PEtabMultistartResult
"""
function run_parameter_estimation(petab_prob; n_multistarts=10)
    println("\n" * "="^60)
    println("  Running multi-start optimization ($n_multistarts starts)")
    println("="^60)

    # Generate starting points: nominal + controlled perturbations
    # This avoids the random Latin Hypercube sampling that generates
    # extreme Arrhenius parameter combinations
    Random.seed!(123)
    nominal = collect(petab_prob.xnominal_transformed)
    lb = collect(petab_prob.lower_bounds)
    ub = collect(petab_prob.upper_bounds)

    all_results = []
    best_fmin = Inf
    best_xmin = nominal

    for i in 1:n_multistarts
        if i == 1
            x0 = copy(nominal)
        else
            # Perturb nominal by ±20% within bounds
            x0 = nominal .+ 0.2 .* (ub .- lb) .* (rand(length(nominal)) .- 0.5)
            x0 = clamp.(x0, lb, ub)
        end

        println("  Start $i/$n_multistarts...")
        try
            res = calibrate(petab_prob, x0, LBFGS())
            fval = res.fmin
            println("    → NLL = $(round(fval, digits=2))")
            if fval < best_fmin
                best_fmin = fval
                best_xmin = res.xmin
            end
            push!(all_results, res)
        catch e
            println("    → Failed (solver error)")
        end
    end

    println("\n  Best NLL: $(round(best_fmin, digits=2))")

    # Return a result-like object compatible with downstream code
    return (xmin = best_xmin, fmin = best_fmin, runs = all_results)
end

"""
    compare_parameters(result, petab_prob)

Create a comparison table of true vs recovered parameters.

Returns: DataFrame with parameter comparison
"""
function compare_parameters(result, petab_prob)
    # True parameter values (from Section 6)
    true_values = Dict(
        "log10_A1" => 5.0,
        "E1"       => 120_000.0,
        "n1"       => 1.0,
        "log10_A2" => 7.0,
        "E2"       => 160_000.0,
        "n2"       => 1.5,
        "m2"       => 0.6,
        "log10_A3" => 9.0,
        "E3"       => 250_000.0,
        "n3"       => 2.0,
        "m3"       => 0.85,
    )

    # Extract recovered values from optimization result
    recovered = Dict(zip(string.(petab_prob.xnames), result.xmin))

    # Build comparison table
    param_names = ["log10_A1", "E1", "n1", "log10_A2", "E2", "n2", "m2",
                   "log10_A3", "E3", "n3", "m3"]
    true_vals = Float64[]
    rec_vals  = Float64[]
    pct_errs  = Float64[]

    for pname in param_names
        tv = true_values[pname]
        rv = recovered[pname]
        pe = abs(rv - tv) / abs(tv) * 100.0
        push!(true_vals, tv)
        push!(rec_vals, rv)
        push!(pct_errs, pe)
    end

    comparison_df = DataFrame(
        Parameter  = param_names,
        True_Value = true_vals,
        Recovered  = round.(rec_vals, sigdigits=5),
        Pct_Error  = round.(pct_errs, digits=2),
    )

    return comparison_df
end

"""
    print_comparison_table(df)

Pretty-print the parameter comparison table.
"""
function print_comparison_table(df)
    println("\n" * "="^70)
    println("  Parameter Estimation Results: True vs Recovered")
    println("="^70)
    @printf("  %-12s  %12s  %12s  %10s\n", "Parameter", "True Value", "Recovered", "% Error")
    println("  " * "-"^50)
    for row in eachrow(df)
        @printf("  %-12s  %12.4f  %12.4f  %9.2f%%\n",
                row.Parameter, row.True_Value, row.Recovered, row.Pct_Error)
    end
    println("="^70)
    println("  Mean % Error: $(round(Statistics.mean(df.Pct_Error), digits=2))%")
    println("="^70)
end


# =============================================================================
# Visualization 
# =============================================================================

function plot_validation(rn, result, petab_prob, clean_data)

    recovered = Dict(zip(petab_prob.xnames, result.xmin))
    opt_params = Dict{Symbol, Float64}()
    for (k, v) in recovered
        opt_params[k] = v
    end
    opt_params[:nu_char] = 0.2

    p = plot(
        xlabel = "Temperature [K]",
        ylabel = "Mass Fraction",
        title  = "TGA Validation: Exp 2 (21% O₂) vs Exp 4 (5% O₂)",
        legend = :bottomleft,
        size   = (900, 550),
        dpi    = 150,
        grid   = true,
        framestyle = :box,
    )

    # =========================
    # EXP 2
    # =========================
    exp2_data = clean_data["Exp2"]

    scatter!(p, exp2_data.temperatures, exp2_data.mass_noisy,
        label="Exp 2 data (21% O₂)",
        color=:blue,
        markersize=1.5,
        alpha=0.25,
        markerstrokewidth=0
    )

    _, temps_opt2, mass_opt2 = simulate_experiment(
        rn, "Exp2", EXPERIMENTS["Exp2"], opt_params; n_points=300
    )

    plot!(p, temps_opt2, mass_opt2,
        label="Exp 2 fit (21% O₂)",
        color=:blue,
        linewidth=2.5
    )

    # =========================
    # EXP 4
    # =========================
    exp4_data = clean_data["Exp4"]

    scatter!(p, exp4_data.temperatures, exp4_data.mass_noisy,
        label="Exp 4 data (5% O₂)",
        color=:red,
        markersize=1.5,
        alpha=0.25,
        markerstrokewidth=0
    )

    _, temps_opt4, mass_opt4 = simulate_experiment(
        rn, "Exp4", EXPERIMENTS["Exp4"], opt_params; n_points=300
    )

    plot!(p, temps_opt4, mass_opt4,
        label="Exp 4 fit (5% O₂)",
        color=:red,
        linewidth=2.5
    )

    savefig(p, "validation_plot.png")
    display(p)

    return p
end


function plot_all_experiments(clean_data)

    colors = Dict(
        "Exp1" => :blue,
        "Exp2" => :green,
        "Exp3" => :orange,
        "Exp4" => :red,
    )

    labels = Dict(
        "Exp1" => "Exp 1: 2.5 K/min, 21% O₂",
        "Exp2" => "Exp 2: 5.0 K/min, 21% O₂",
        "Exp3" => "Exp 3: 10.0 K/min, 21% O₂",
        "Exp4" => "Exp 4: 5.0 K/min, 5% O₂",
    )

    p = plot(
        xlabel = "Temperature [K]",
        ylabel = "Mass Fraction",
        title  = "Synthetic TGA Data — All Experiments",
        legend = :bottomleft,
        size   = (900, 550),
        grid   = true,
        framestyle = :box
    )

    for exp_id in ["Exp1","Exp2","Exp3","Exp4"]
        data = clean_data[exp_id]

        # DATA (dots)
        scatter!(p, data.temperatures, data.mass_noisy,
            label = labels[exp_id] * " (noisy)",
            color = colors[exp_id],
            markersize = 1.5,
            alpha = 0.25,
            markerstrokewidth = 0
        )

        # TRUE (line)
        plot!(p, data.temperatures, data.mass_clean,
            label = labels[exp_id] * " (true)",
            color = colors[exp_id],
            linewidth = 2.5
        )
    end

    savefig(p, "all_experiments.png")
    display(p)

    return p
end


# =============================================================================
#  MAIN EXECUTION
# =============================================================================

println("\n===== RUNNING FULL PIPELINE =====")

rn = build_cfrp_reaction_system()

measurements_df, clean_data = generate_all_synthetic_data(rn)

petab_prob = setup_petab_model(rn, measurements_df)

result = run_parameter_estimation(petab_prob)

comparison_df = compare_parameters(result, petab_prob)

print_comparison_table(comparison_df)

plot_validation(rn, result, petab_prob, clean_data)
plot_all_experiments(clean_data)

println("\n✅ ALL RESULTS SAVED IN: C:/Users/user/results")
