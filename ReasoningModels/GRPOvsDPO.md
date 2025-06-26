

## 🧠 Part 1: **GRPO vs DPO – Key Differences**

| Aspect                      | GRPO (Guided Reward Policy Optimization)    | DPO (Direct Preference Optimization)                      |
| --------------------------- | ------------------------------------------- | --------------------------------------------------------- |
| **Input**                   | Reward model trained on scalar rewards      | Pairwise preferences (e.g., A > B)                        |
| **Optimization Target**     | Maximize reward with KL-penalized objective | Maximize preference likelihood                            |
| **Anchor Policy**           | Yes, used explicitly for KL regularization  | Yes, used as reference in likelihood ratio                |
| **KL Regularization**       | Yes (explicit KL penalty)                   | Yes (implicit via loss formulation)                       |
| **Training Stability**      | High, guided by anchor policy               | High, due to direct optimization                          |
| **Output**                  | Optimized policy $\pi_\theta$               | Optimized policy $\pi_\theta$                             |
| **Mathematical Base**       | Policy gradient (REINFORCE-style)           | Binary classification (cross-entropy on preference pairs) |
| **Sample Efficiency**       | Moderate to High                            | Very High (no rollouts needed)                            |
| **Reward Model Dependency** | Required                                    | Indirect (used to rank only)                              |

### 🔁 Summary:

* **GRPO**: RL-style training, anchored with KL to a supervised model.
* **DPO**: Directly optimizes preferences with a contrastive loss — no explicit reward modeling needed.

---

