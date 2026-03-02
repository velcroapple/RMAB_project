from patient_scheduler import Patient, PatientScheduler, WhittleCalculator
import numpy as np
import matplotlib.pyplot as plt
import csv
import json
import os

def setup_whittle_indices(patients, recompute=True, cache_path="whittle_cache.json"):
    if recompute:
        for p in patients:
            p.indices = WhittleCalculator.get_indices(p.P0, p.P1, p.R, p.beta)
        save_whittle_cache(patients, cache_path)
    else:
        if not load_whittle_cache(patients, cache_path):
            setup_whittle_indices(patients, recompute=True, cache_path=cache_path)

def save_whittle_cache(patients, filepath="whittle_cache.json", verbose=False):
    cache = {str(p.id): {"indices": p.indices} for p in patients}
    with open(filepath, "w") as f:
        json.dump(cache, f, indent=2)
    if verbose:
        print(f"Whittle cache saved for {len(patients)} patients.")

def load_whittle_cache(patients, filepath="whittle_cache.json", verbose=False):
    if not os.path.exists(filepath):
        if verbose:
            print("No cache found, recomputing...")
        return False
    with open(filepath, "r") as f:
        cache = json.load(f)
    for p in patients:
        p.indices = {int(k): v for k, v in cache[str(p.id)]["indices"].items()}
    if verbose:
        print("Whittle cache loaded.")
    return True

def load_configs(filepath="configs.csv"):
    configs = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs.append({
                "p0": json.loads(row["p0"]),
                "p1": json.loads(row["p1"]),
                "r":  json.loads(row["r"]),
            })
    return configs

def plot_policy_matrix_on_ax(ax, policy_matrix, lambdas, title, state_labels=None):
    K, J = policy_matrix.shape

    if state_labels is None:
        state_labels = [f's={i}' for i in range(K)]

    im = ax.imshow(
        policy_matrix,
        cmap='coolwarm',
        aspect='auto',
        interpolation='none',
        vmin=0,
        vmax=1
    )

    ax.set_yticks(range(K))
    ax.set_yticklabels(state_labels)

    tick_indices = range(0, J, max(1, J // 5))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f'{lambdas[i]:.2f}' for i in tick_indices], rotation=45)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('lambda')
    ax.set_ylabel('State')

    ax.set_xticks(np.arange(J) - 0.5, minor=True)
    ax.set_yticks(np.arange(K) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=0.4)

    return im

def verify_and_plot_all_patients(patients, plot_limit=8, recompute=True, cache_path="whittle_cache.json"):
    setup_whittle_indices(patients, recompute=recompute, cache_path=cache_path)

    plot_n = min(plot_limit, len(patients))
    cols = 4
    rows = int(np.ceil(plot_n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    axes = axes.flatten()

    for i, patient in enumerate(patients[:plot_n]):
        policy_matrix, lambdas = WhittleCalculator.compute_policy_matrix(
            patient.P0, patient.P1, patient.R, patient.beta
        )
        plot_policy_matrix_on_ax(axes[i], policy_matrix, lambdas,
                                  title=f"Patient {patient.id}", state_labels=["Healthy", "Sick"])

    for j in range(plot_n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Policy Matrices Across Patients", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def run_simulation(num_days=200, beta=0.95, k=10, recompute_whittle=False, cache_path="whittle_cache.json", fairness_limit=50):
    patients = [Patient(i, c['p0'], c['p1'], c['r'], beta=beta) for i, c in enumerate(configs)]
    scheduler = PatientScheduler(patients, k=k, beta=beta,fairness_limit=fairness_limit)

    verify_and_plot_all_patients(patients, recompute=recompute_whittle, cache_path=cache_path)

    print("WHITTLE INDICES")
    print(f"{'ID':<4} | {'Whittle (Healthy)':<18} | {'Whittle (Sick)':<15}")
    print("-" * 60)
    for p in patients:
        print(f"{p.id:<4} | {p.indices[0]:.4f}            | {p.indices[1]:.4f}")

    total_reward = 0
    for day in range(1, num_days + 1):
        print(f"\nDAY {day}\n")
        
        day_log, step_reward = scheduler.step()
        total_reward += step_reward
        
        day_log.sort(key=lambda x: (x['action'] != 'Scheduled', x['id']))
        if day % 50 == 0:
            print(f"\n{'Patient':<10} | {'Action':<12} | {'State':<10} | {'Belief':<10} | {'Reward':<6}")
            print("-" * 60)
            
            for entry in day_log:
                print(f"{entry['id']:<10} | {entry['action']:<12} | "
                    f"{entry['current_state']:<10} | "
                    f"{entry['belief']:.4f}     | "
                    f"{entry['reward']:.2f}")
            
            print(f"\nDay {day} Reward: {step_reward:.2f}")
            print(f"Cumulative Reward: {total_reward:.2f}")
            print(f"Discounted Cumulative Reward: {scheduler.cumulative_reward:.2f}")

    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print(f"Total Days: {num_days}")
    print(f"Total Undiscounted Reward: {total_reward:.2f}")
    print(f"Total Discounted Reward: {scheduler.cumulative_reward:.2f}")
    print(f"Average Reward per Day: {total_reward/num_days:.2f}")
    
    waits, violated = scheduler.longest_waits(fairness_limit=50)
    print(f"\nFairness Check (Longest Wait >= 50):")
    if violated:
        print(f"total patients with fairness violation: {len(violated)} out of {len(patients)}")
        print(f"patients who never got scheduled: {len([w['id'] for w in waits if w['longest_wait'] >= num_days + 1])}")
        for idx in violated:
            print(f"Patient {idx} - Longest Wait: {waits[idx]['longest_wait']} timesteps")
    else:
        print("No patients violated the fairness constraint.")

    print(f"\n{'Patient':<10} | {'Final State':<12} | {'Final Belief':<15}")
    print("-" * 45)
    for p in patients:
        state = "Healthy" if p.true_state == 0 else "Sick"
        print(f"{p.id:<10} | {state:<12} | {p.belief_sick:.4f}")

def run_comparison(num_simulations=10, num_days=100, beta=0.95, k=10, cache_path="whittle_cache.json", 
                   print_every=10, fairness_limit=50, recompute=False):
    print("\n" + "=" * 60)
    print("RUNNING COMPARISON OVER MULTIPLE SIMULATIONS")
    print("=" * 60)
    
    rewards = []
    all_violations = []
    never_scheduled = []
    treatment_scores = []
    for sim in range(num_simulations):
        patients = [Patient(i, c['p0'], c['p1'], c['r'], beta=beta) for i, c in enumerate(configs)]
        setup_whittle_indices(patients, recompute=recompute, cache_path=cache_path)
        recompute=False  # only recompute for the first simulation, then load from cache for subsequent sims
        scheduler = PatientScheduler(patients, k=k, beta=beta,fairness_limit=fairness_limit)
        
        for _ in range(num_days):
            scheduler.step()
        
        rewards.append(scheduler.cumulative_reward)
        treatment_scores.append(scheduler.treatment_score / num_days)
        
        waits, violated = scheduler.longest_waits(fairness_limit=fairness_limit)
        all_violations.append(len(violated))
        never_scheduled.append(len([w for w in waits if w['longest_wait'] >= num_days + 1]))

        if (sim + 1) % print_every == 0:
            print(f"Sim {sim + 1}/{num_simulations} | Reward: {scheduler.cumulative_reward:.2f} | Violations: {len(violated)}")

    print(f"\nSimulations: {num_simulations}")
    print(f"Days per simulation: {num_days}")
    print(f"Average Discounted Reward: {np.mean(rewards):.2f}")
    print(f"Std Dev: {np.std(rewards):.2f}")
    print(f"Min: {np.min(rewards):.2f}")
    print(f"Max: {np.max(rewards):.2f}")
    print(f"\nFairness (wait >= {fairness_limit}):")
    print(f"  Avg violations per sim: {np.mean(all_violations):.2f}")
    print(f"  Sims with any violation: {sum(1 for v in all_violations if v > 0)}/{num_simulations}")
    print(f"  Avg never scheduled per sim: {np.mean(never_scheduled):.2f}")
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(rewards)
    axes[0].set_title("Discounted Reward per Sim")
    axes[0].set_xlabel("Simulation")
    axes[0].set_ylabel("Reward")

    axes[1].plot(treatment_scores)
    axes[1].set_title("Treatment Efficiency per Sim")
    axes[1].set_xlabel("Simulation")
    axes[1].set_ylabel("Score / Day")
    axes[1].axhline(0, color='red', linestyle='--')  # zero line

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    configs = load_configs(r"C:\Users\avish\OneDrive\Documents\AVISHA SEM 4\MULTI ARMED BANDIT PROJECT\configs.csv")
    #run_simulation(num_days=200, k=10, beta=0.85, recompute_whittle=False, cache_path="whittle_cache.json")
    
    run_comparison(num_simulations=50, num_days=200, beta=0.95,k=10, recompute=True, cache_path="whittle_cache.json", fairness_limit=50)