"""
tri-hybrid-sim: Cyber-Physical Microgrid Dynamic Simulator
Models frequency stability and ramp-rate transients under AI cluster step loads.
"""

import argparse
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class MicrogridConfig:
    # Grid & Physical Constants
    nominal_freq: float = 60.0       # Hz
    inertia_h: float = 3.5           # Turbine inertia constant H (seconds)
    ufls_trip_freq: float = 59.2      # Under-Frequency Load Shedding threshold (Hz)
    
    # Turbine Dynamics (Aeroderivative Gas / SMR surrogate)
    turbine_capacity_mw: float = 60.0
    turbine_max_ramp_mw_s: float = (0.10 * 60.0) / 60.0  # 10%/min = 0.10 MW/s
    turbine_gov_tau: float = 0.40    # Mechanical governor valve time constant (s)

    # Inverter-Based BESS (Battery Energy Storage System)
    bess_capacity_mwh: float = 15.0
    bess_max_power_mw: float = 25.0
    bess_response_tau: float = 0.020 # 20 ms inverter injection delay


def run_simulation(config: MicrogridConfig, enable_bess: bool, dt: float = 0.001, t_end: float = 12.0):
    time = np.arange(0, t_end, dt)
    n = len(time)

    # State vectors
    p_demand = np.full(n, 15.0)       # 15 MW idle baseline
    p_demand[time >= 1.5] = 50.0      # 35 MW collective burst at t = 1.5s

    p_turbine = np.zeros(n)
    p_bess = np.zeros(n)
    freq = np.zeros(n)

    # Initial steady-state conditions
    p_turbine[0] = 15.0
    p_bess[0] = 0.0
    freq[0] = config.nominal_freq
    tripped = False
    trip_time = None

    for i in range(1, n):
        # 1. BESS Fast-Frequency Actuation (Virtual Inertia / Droop Control)
        if enable_bess and not tripped:
            freq_deviation = config.nominal_freq - freq[i-1]
            # Fast proportional response to frequency deviation
            p_bess_target = np.clip(freq_deviation * 80.0, -config.bess_max_power_mw, config.bess_max_power_mw)
            # Low-pass filter representing inverter response delay
            p_bess[i] = p_bess[i-1] + (p_bess_target - p_bess[i-1]) * (dt / config.bess_response_tau)
        else:
            p_bess[i] = 0.0

        # 2. Turbine Governor & Ramp-Rate Limiter
        p_target = p_demand[i-1] - p_bess[i]
        dp_raw = (p_target - p_turbine[i-1]) * (dt / config.turbine_gov_tau)
        dp_clamped = np.clip(dp_raw, -config.turbine_max_ramp_mw_s * dt, config.turbine_max_ramp_mw_s * dt)
        p_turbine[i] = p_turbine[i-1] + dp_clamped

        # 3. Dynamic Swing Equation: d(omega)/dt = (P_mech + P_bess - P_elec) / (2 * H)
        net_imbalance_pu = (p_turbine[i] + p_bess[i] - p_demand[i]) / config.turbine_capacity_mw
        df_dt = (net_imbalance_pu / (2 * config.inertia_h)) * config.nominal_freq
        freq[i] = freq[i-1] + df_dt * dt

        # 4. Under-Frequency Protection Relay Trip
        if freq[i] <= config.ufls_trip_freq and not tripped:
            tripped = True
            trip_time = time[i]
            p_turbine[i:] = 0.0
            p_bess[i:] = 0.0
            freq[i:] = freq[i]
            break

    return {
        "time": time,
        "demand": p_demand,
        "turbine": p_turbine,
        "bess": p_bess,
        "freq": freq,
        "tripped": tripped,
        "trip_time": trip_time
    }


def generate_comparison_plot(unmitigated, mitigated, output_path: str = "simulation_benchmark.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Power Dispatch Subplot
    ax1.plot(unmitigated["time"], unmitigated["demand"], 'k--', label="AI Compute Demand (50 MW Step)", alpha=0.7)
    ax1.plot(unmitigated["time"], unmitigated["turbine"], 'r-', label="Turbine (Unmitigated Baseline)", linewidth=1.8)
    ax1.plot(mitigated["time"], mitigated["turbine"], 'g-', label="Turbine (Governed + BESS)", linewidth=2.0)
    ax1.plot(mitigated["time"], mitigated["bess"], 'b:', label="BESS Transient Injection (MW)", linewidth=1.8)
    
    if unmitigated["tripped"]:
        ax1.axvline(unmitigated["trip_time"], color='red', linestyle='--', alpha=0.8, 
                    label=f"Unmitigated Trip ({unmitigated['trip_time']:.2f}s)")

    ax1.set_ylabel("Active Power (MW)")
    ax1.set_title("Tri-Hybrid Microgrid Dynamics: AI Step Load Transient Response", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Frequency Subplot
    ax2.plot(unmitigated["time"], unmitigated["freq"], 'r-', label="Frequency: Unmitigated (Trip)", linewidth=1.8)
    ax2.plot(mitigated["time"], mitigated["freq"], 'g-', label="Frequency: Governed (Stable)", linewidth=2.0)
    ax2.axhline(59.2, color='darkred', linestyle='--', label="UFLS Limit (59.2 Hz)")
    ax2.axhline(60.0, color='gray', linestyle=':', alpha=0.6)
    
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylim(58.8, 60.2)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[SUCCESS] Benchmark visualization exported to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tri-Hybrid AI Microgrid Dynamic Simulator")
    parser.add_argument("--save-plot", type=str, default="benchmark_comparison.png", help="Filename for output plot")
    args = parser.parse_args()

    cfg = MicrogridConfig()

    print("[*] Running Unmitigated Baseline Simulation...")
    res_unmitigated = run_simulation(cfg, enable_bess=False)

    print("[*] Running Governed Tri-Hybrid Simulation (BESS Enabled)...")
    res_mitigated = run_simulation(cfg, enable_bess=True)

    if res_unmitigated["tripped"]:
        print(f" -> Baseline: System CRASHED at t = {res_unmitigated['trip_time']:.3f} s (Frequency fell below {cfg.ufls_trip_freq} Hz).")
    if not res_mitigated["tripped"]:
        min_freq = np.min(res_mitigated["freq"])
        print(f" -> Governed: System STABLE. Frequency nadir held safely at {min_freq:.3f} Hz.")

    generate_comparison_plot(res_unmitigated, res_mitigated, output_path=args.save_plot)
