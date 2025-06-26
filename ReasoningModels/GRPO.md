Sure! Let's walk through **GRPO (Guided Reward Policy Optimization)** — an algorithm designed to improve the sample efficiency and stability of training **Reinforcement Learning from Human Feedback (RLHF)** systems, such as **Large Language Models (LLMs)**.

---

## 🚀 Overview of GRPO

**GRPO** stands for **Guided Reward Policy Optimization**.
It is proposed as a better alternative to **PPO (Proximal Policy Optimization)**, the default RL method used in RLHF pipelines like ChatGPT fine-tuning.

### 🔍 Why GRPO?

* Traditional **RLHF** setups use **PPO** to fine-tune models using a learned reward model.
* PPO has some downsides:

  * Sample inefficiency.
  * Instability due to exploration noise.
  * Difficulty scaling to very large models.
* **GRPO** aims to **stabilize** and **guide** the learning process using **an anchor policy**.

---

## 📘 Key Concepts in GRPO

### 1. **Anchor Policy (πₐ)**

* A fixed reference policy, usually the supervised fine-tuned model (SFT).
* It acts as a **guide** for the learning process.
* The final policy is **regularized** to stay close to the anchor policy.

### 2. **Reward Model (R(x))**

* Trained to mimic human preference scores.
* Provides the **scalar reward** for each output.

### 3. **KL Regularization**

* KL-divergence is used to **penalize deviation** from the anchor policy.
* This keeps the learned policy aligned with what humans expect (based on SFT).

---

## 🧠 GRPO Algorithm Steps

Let’s break the training loop of GRPO:

### Step 1: **Collect Samples**

Generate responses using the current policy $\pi_\theta$ (learned policy), conditioned on prompts.

### Step 2: **Compute Rewards**

Use the reward model $r(x, y)$ to evaluate the responses.

Total reward includes:

$$
R_{\text{total}} = r(x, y) - \beta \cdot \text{KL}(\pi_\theta || \pi_a)
$$

Where:

* $\pi_\theta$: current policy
* $\pi_a$: anchor policy
* $\beta$: KL coefficient (controls regularization strength)

### Step 3: **Compute Policy Gradient**

Instead of computing the standard policy gradient using PPO, GRPO uses a **guided advantage**:

$$
\mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot (R - b(s)) \right]
$$

Where:

* $b(s)$: baseline (value function or mean reward)
* The KL penalty shifts the gradient toward staying close to the anchor policy.

### Step 4: **Update the Policy**

Apply gradient ascent to improve $\pi_\theta$, while keeping it close to $\pi_a$.

---

## 💡 GRPO vs PPO

| Feature               | PPO                             | GRPO                                 |
| --------------------- | ------------------------------- | ------------------------------------ |
| Reference policy      | Previous policy                 | Anchor policy (usually SFT)          |
| KL penalty direction  | From new to old policy          | From current policy to anchor policy |
| Stability             | Can be unstable on large models | More stable due to anchor guidance   |
| Sample efficiency     | Moderate                        | Better                               |
| Alignment with humans | Indirect                        | Direct via anchor                    |

---

## 📈 Mathematical Formulation

The GRPO objective is:

$$
\mathcal{L}_{\text{GRPO}} = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) - \beta \cdot \text{KL}(\pi_\theta(\cdot|x) || \pi_a(\cdot|x)) \right]
$$

Then, optimize this using standard policy gradient methods.

---

## 🧪 Intuition

* GRPO helps **guide the policy toward more human-aligned behaviors**.
* It avoids the collapse or over-exploration often seen in PPO-based training.
* It acts like “**you can improve, but don’t stray too far from your teacher**.”

---

## 🛠️ Practical Use Case: LLM Fine-Tuning

* **SFT model** becomes the anchor.
* **Reward model** trained on preference rankings from humans.
* **GRPO** helps fine-tune the model so that it improves helpfulness and harmlessness while not diverging from supervised knowledge.

---

## 🧵 In Summary

**GRPO = Supervised + Reward Optimization + Regularization**

* Uses an **anchor policy** to avoid overfitting to reward model quirks.
* Improves stability and alignment over PPO.
* Ideal for fine-tuning LLMs with human feedback.

---

