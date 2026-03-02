import numpy as np
import matplotlib.pyplot as plt

class WhittleCalculator:
    """Computes Whittle Indices for a specific set of dynamics using a subsidy grid."""
    
    @staticmethod
    def value_iteration(P0, P1, R, beta, lambda_subsidy, states=[0, 1], tol=1e-4):
        """
        Find the optimal policy for a given subsidy lambda using value iteration.
        Inputs:
            P0, P1: Transition matrices for passive and active actions
            R: Reward matrix where R[s, a] is reward for state s and action a
            beta: Discount factor
            lambda_subsidy: The subsidy added to the reward of passive action
            states: 0: Healthy, 1: Sick
            tol: Convergence tolerance for value iteration

        Returns:
            bool: whether it is optimal to take active action (1) over passive action (0) for each state under the given subsidy
        """
        v = np.zeros(len(states))
        for _ in range(100):
            v_old = v.copy()
            q0 = R[:, 0] + lambda_subsidy + beta * P0 @ v
            q1 = R[:, 1] + beta * P1 @ v
            v = np.maximum(q0, q1)
            if np.max(np.abs(v - v_old)) < tol:
                break
        return (q1 >= q0).astype(int)
    
    @staticmethod
    def get_indices(P0, P1, R, beta):

        """
        Compute Whittle indices for states 0 and 1 by finding the subsidy lambda where policy switches.
        """
        lambdas = np.linspace(-1, 3, 200)
        indices = {0: -1.0, 1: -1.0}
        for s in [0, 1]:
            for l in lambdas:
                policy = WhittleCalculator.value_iteration(P0, P1, R, beta, l)
                if policy[s] == 0:
                    indices[s] = l
                    break
        return indices
    
    @staticmethod
    def compute_policy_matrix(P0, P1, R, beta, lambda_grid=None):
        """
        Compute the policy matrix phi as described in the paper (Algorithm 1).
        
        Returns:
            policy_matrix: K x J matrix where entry [s, j] is the optimal action
                          for state s under subsidy lambda_j
            lambdas: The grid of subsidy values used
        """
        if lambda_grid is None:
            lambdas = np.linspace(-1, 3, 50)  # J = 50 subsidy values
        else:
            lambdas = lambda_grid
        
        K = len(R)  # Number of states
        J = len(lambdas)  # Number of subsidy values
        
        policy_matrix = np.zeros((K, J), dtype=int)
        
        for j, lam in enumerate(lambdas):
            policy = WhittleCalculator.value_iteration(P0, P1, R, beta, lam)
            policy_matrix[:, j] = policy
        
        return policy_matrix, lambdas
    
    @staticmethod
    def verify_indexability(policy_matrix, lambdas, verbose=False):
        """
        Verify indexability using the policy matrix approach from Section III-B.
        
        For each state s, check if policy has single threshold structure:
        - Once policy becomes 0 (passive), it should never go back to 1 (active)
        
        Returns:
            is_indexable: bool
            indices: dict mapping state to its Whittle index (or None if non-indexable)
        """
        K = policy_matrix.shape[0]
        is_indexable = True
        indices = {}
        
        for s in range(K):
            policy_s = policy_matrix[s, :]
            
            # Check for single threshold structure
            found_passive = False
            threshold_lambda = None
            state_indexable = True
            
            for j, action in enumerate(policy_s):
                if action == 0:  # Passive
                    if not found_passive:
                        found_passive = True
                        threshold_lambda = lambdas[j]
                elif action == 1 and found_passive:  # Active after passive
                    state_indexable = False
                    is_indexable = False
                    if verbose:
                        print(f"State {s} is NON-INDEXABLE: "
                              f"policy goes back to active at lambda={lambdas[j]:.3f}")
                    break
            
            if state_indexable:
                indices[s] = threshold_lambda if threshold_lambda is not None else -np.inf
                if verbose:
                    if threshold_lambda is not None:
                        print(f"State {s} is indexable with W({s}) = {threshold_lambda:.4f}")
                    else:
                        print(f"State {s} is indexable (always active)")
            else:
                indices[s] = None
        
        return is_indexable, indices
    
    @staticmethod
    def plot_policy_matrix(policy_matrix, lambdas, title="Policy Matrix phi", 
                          state_labels=None):
        """
        Visualize the policy matrix similar to Figure 1, 2, 3 in the paper.
        """
        K, J = policy_matrix.shape
        
        if state_labels is None:
            state_labels = [f's={i}' for i in range(K)]
        
        fig, ax = plt.subplots(figsize=(12, max(4, K * 0.8)))
        
        # Create custom colormap: 0 (passive) = blue, 1 (active) = orange
        cmap = plt.cm.colors.ListedColormap(['#4A90E2', '#F5A623'])
        
        im = ax.imshow(policy_matrix, cmap=cmap, aspect='auto', 
                      interpolation='none', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_yticks(range(K))
        ax.set_yticklabels(state_labels)
        
        # Show every 5th lambda value
        tick_indices = range(0, J, max(1, J // 10))
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f'{lambdas[i]:.2f}' for i in tick_indices], rotation=45)
        
        ax.set_xlabel('Subsidy lambda', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Passive (0)', 'Active (1)'])
        
        # Add grid
        ax.set_xticks(np.arange(J) - 0.5, minor=True)
        ax.set_yticks(np.arange(K) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=0.5)
        
        plt.tight_layout()
        return fig, ax

class Patient:
    def __init__(self, patient_id, p0, p1, rewards, beta=0.9):
        self.id = patient_id
        self.P0 = np.array(p0)
        self.P1 = np.array(p1)
        self.R = np.array(rewards)
        self.beta = beta
        self.longest_wait = 0
        self.last_iter = -1
        self.indices = None  # set by setup_whittle_indices
        self.true_state = np.random.choice([0, 1])
        self.belief_sick = 0.5


class PatientScheduler:
        
    def __init__(self, patients, k=10, beta=0.9, fairness_limit=50):
        self.patients = patients
        self.k = k
        self.beta = beta
        self.fairness_limit = fairness_limit
        self.cumulative_reward = 0
        self.timestep = 0
        self.overdue = set()  # patients who have exceeded fairness limit
        self.treatment_score=0
        
    def longest_waits(self,fairness_limit=20):
        waits=[]
        violated=[]
        for i,p in enumerate(self.patients):
            p.longest_wait=max(p.longest_wait,self.timestep-p.last_iter)
            if p.longest_wait>=fairness_limit:
                violated.append(i)
            waits.append({"id":i,
                         "longest_wait":p.longest_wait
                         })


        return waits,violated

    def step(self):
        scores = []
        for p in self.patients:
            wait = self.timestep - p.last_iter
            if wait >= self.fairness_limit:
                self.overdue.add(p.id)
                scores.append(float('inf')+wait)  # prioritize overdue patients
            else:
                expected_w = (1 - p.belief_sick) * p.indices[0] + p.belief_sick * p.indices[1]
                scores.append(expected_w)

        if len(self.overdue) > self.k:
            print(f"WARNING: {len(self.overdue)} overdue patients but only k={self.k} slots at timestep {self.timestep}")

        scheduled_indices = np.argsort(scores)[-self.k:]

        results = []
        step_reward = 0

        for i, p in enumerate(self.patients):
            if i in scheduled_indices:
                reward = p.R[p.true_state, 1]
                step_reward += reward
                p.true_state = np.random.choice([0, 1], p=p.P1[p.true_state])
                p.belief_sick = float(p.true_state)
                action_taken = "Scheduled"
                p.longest_wait = max(p.longest_wait, self.timestep - p.last_iter)
                p.last_iter = self.timestep
                self.overdue.discard(p.id)  # remove from overdue once scheduled
            else:
                reward = p.R[p.true_state, 0]
                step_reward += reward
                p.true_state = np.random.choice([0, 1], p=p.P0[p.true_state])
                p.belief_sick = ((1 - p.belief_sick) * p.P0[0, 1] + p.belief_sick * p.P0[1, 1])
                action_taken = "Wait"

            if p.true_state == 1:
                if action_taken == "Scheduled":
                    self.treatment_score += 1
                else:
                    self.treatment_score -= 1
            
            results.append({
                "id": i,
                "action": action_taken,
                "current_state": "Sick" if p.true_state == 1 else "Healthy",
                "belief": p.belief_sick,
                "reward": reward,
                "time since last visit": self.timestep - p.last_iter
            })

        self.cumulative_reward += (self.beta ** self.timestep) * step_reward
        self.timestep += 1

        return results, step_reward
