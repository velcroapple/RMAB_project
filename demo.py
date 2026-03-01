from patient_scheduler import Patient, PatientScheduler, WhittleCalculator
import numpy as np
import matplotlib.pyplot as plt

# --- 8 PATIENTS WITH DIFFERENT DYNAMICS ---
# p0: Transitions if Wait | p1: Transitions if Scheduled
# Rewards: [[H_wait, H_sched], [S_wait, S_sched]]
configs = [
    {"p0": [[0.9, 0.1], [0.3, 0.7]], "p1": [[0.9, 0.1], [0.8, 0.2]], "r": [[1, 0.5], [0, 1]]},  
    {"p0": [[0.8, 0.2], [0.1, 0.9]], "p1": [[0.9, 0.1], [0.7, 0.3]], "r": [[1, 0.5], [0, 1]]},  
    {"p0": [[0.9, 0.1], [0.1, 0.9]], "p1": [[0.99, 0.01], [0.8, 0.2]], "r": [[1, 0.5], [0, 1]]}, # Recoverer
    {"p0": [[0.8, 0.2], [0.2, 0.8]], "p1": [[0.9, 0.1], [0.4, 0.6]], "r": [[1, 0.5], [0, 1]]}, 
    {"p0": [[0.65, 0.35], [0.1, 0.9]], "p1": [[0.9, 0.1], [0.45, 0.55]], "r": [[1, 0.5], [0, 1]]},  # Fragile
    {"p0": [[0.95, 0.05], [0.05, 0.95]], "p1": [[0.99, 0.01], [0.9, 0.1]], "r": [[1, 0.5], [0, 1]]}, 
    {"p0": [[0.7, 0.3], [0.5, 0.5]], "p1": [[0.8, 0.2], [0.7, 0.3]], "r": [[1, 0.5], [0, 1]]},  # High-Risk
    {"p0": [[0.9, 0.1], [0.2, 0.8]], "p1": [[0.95, 0.05], [0.6, 0.4]],"r": [[1, 0.5], [0, 1]]}, 
]

def plot_policy_matrix_on_ax(ax, policy_matrix, lambdas, title, state_labels=None):
    K, J = policy_matrix.shape

    if state_labels is None:
        state_labels = [f's={i}' for i in range(K)]

    cmap = plt.cm.colors.ListedColormap(['#4A90E2', '#F5A623'])

    im = ax.imshow(
        policy_matrix,
        cmap=cmap,
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


def verify_and_plot_all_patients(patients):
    n = len(patients)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4 * cols, 3 * rows),
        squeeze=False
    )

    axes = axes.flatten()
    last_im = None

    for i, patient in enumerate(patients):
        print(f"\nVerifying indexability for Patient {patient.id}")
        print("-" * 50)

        policy_matrix, lambdas = WhittleCalculator.compute_policy_matrix(
            patient.P0, patient.P1, patient.R, patient.beta
        )

        is_indexable, _ = WhittleCalculator.verify_indexability(
            policy_matrix, lambdas, verbose=True
        )

        if is_indexable:
            last_im = plot_policy_matrix_on_ax(
                axes[i],
                policy_matrix,
                lambdas,
                title=f"Patient {patient.id}",
                state_labels=["Healthy", "Sick"]
            )
        else:
            axes[i].set_title(f"Patient {patient.id} (Non-indexable)")
            axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Policy Matrices Across Patients", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def run_simulation(num_days=10, beta=0.9):
    # Instantiate Patients with consistent beta
    patients = [Patient(i, c['p0'], c['p1'], c['r'], beta=beta) for i, c in enumerate(configs)]
    scheduler = PatientScheduler(patients, k=3, beta=beta)
    # --- VERIFY INDEXABILITY AND PLOT POLICIES ---
    verify_and_plot_all_patients(patients)
    # Display Whittle Indices

    print("WHITTLE INDICES")
    
    print(f"{'ID':<4} | {'Whittle (Healthy)':<18} | {'Whittle (Sick)':<15}")
    print("-" * 60)
    for p in patients:
        print(f"{p.id:<4} | {p.indices[0]:.4f}            | {p.indices[1]:.4f}")
    
    # Simulation loop
    total_reward = 0
    for day in range(1, num_days + 1):
        print(f"\nDAY {day}\n")
        
        # Get results and rewards from step
        day_log, step_reward = scheduler.step()
        total_reward += step_reward
        
        # Display scheduled patients first, then waiting
        day_log.sort(key=lambda x: (x['action'] != 'Scheduled', x['id']))
        
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
    
    # Final Summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print(f"Total Days: {num_days}")
    print(f"Total Undiscounted Reward: {total_reward:.2f}")
    print(f"Total Discounted Reward: {scheduler.cumulative_reward:.2f}")
    print(f"Average Reward per Day: {total_reward/num_days:.2f}")
    
    # Patient statistics
    print(f"\n{'Patient':<10} | {'Final State':<12} | {'Final Belief':<15}")
    print("-" * 45)
    for p in patients:
        state = "Healthy" if p.true_state == 0 else "Sick"
        print(f"{p.id:<10} | {state:<12} | {p.belief_sick:.4f}")

def run_comparison(num_simulations=100, num_days=50):
    """Run multiple simulations to get average performance"""
    print("\n" + "=" * 60)
    print("RUNNING COMPARISON OVER MULTIPLE SIMULATIONS")
    print("=" * 60)
    
    rewards = []
    
    for sim in range(num_simulations):
        patients = [Patient(i, c['p0'], c['p1'], c['r']) for i, c in enumerate(configs)]
        scheduler = PatientScheduler(patients, k=3)
        
        for day in range(num_days):
            scheduler.step()
        
        rewards.append(scheduler.cumulative_reward)
    
    print(f"\nSimulations: {num_simulations}")
    print(f"Days per simulation: {num_days}")
    print(f"Average Discounted Reward: {np.mean(rewards):.2f}")
    print(f"Std Dev: {np.std(rewards):.2f}")
    print(f"Min: {np.min(rewards):.2f}")
    print(f"Max: {np.max(rewards):.2f}")

if __name__ == "__main__":
    # Single detailed simulation
    run_simulation(num_days=10)
    
    #Optional: Run comparison
    #run_comparison(num_simulations=50, num_days=20)
