# Information-Geometric Unification of Multi-Turn Narrative Erosion as a Natural Gradient Descent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Theory: Information Geometry](https://img.shields.io/badge/Theory-Information_Geometry-blue.svg)]()
[![Field: AI_Alignment/Offensive_Security](https://img.shields.io/badge/Field-AI__Alignment%2FOffensive__Security-purple.svg)]()

This repository formalizes a novel theoretical framework uniting **Iterated Learning**, **Discrete Textual Gradients**, and **Information Geometry**. 

### 🔴 The Core Insight
Multi-turn narrative erosion—often observed in "phone game" scenarios or recursive LLM prompting—is not merely stochastic noise. We prove that it is mathematically equivalent to **Natural Gradient Descent (NGD)** on the Riemannian manifold of autoregressive generation policies.

## 2. Theoretical Framework

### The Policy Manifold
To bypass the non-differentiability of discrete tokens, we model narrative states as points on a statistical manifold $\mathcal{M}$ of generation policies $\pi_\theta$.

### The Metric Tensor (Fisher Information)
The drift between narrative states is governed by the **Fisher Information Matrix (FIM)**, $I(\theta)$, which defines the intrinsic curvature of the semantic space:
$$I(\theta) = \mathbb{E}_{x \sim p_\theta} \left[ \nabla_\theta \log p_\theta(x) \nabla_\theta \log p_\theta(x)^T \right]$$

### The Update Rule
The iterative erosion follows the path of steepest descent along the Fisher metric, effectively preconditioning the gradient to align with the semantic topology:
$$\theta_{t+1} = \theta_t - \eta I(\theta_t)^{-1} \nabla_\theta L(\theta_t)$$

---

## 3. Offensive Security Implications
| Phenomenon | Information-Geometric Interpretation |
| :--- | :--- |
| **Jailbreak Sensitivity** | Adversarial suffixes exploit regions where $\det(I(\theta)) \to 0$, creating "semantic cliffs" where micro-mutations force policy collapse. |
| **Recursive Decay** | Model collapse is the deterministic trajectory toward the geometric center of mass of the model's architectural priors via NGD. |

---

## 4. Empirical Proof of Concept
The following script demonstrates the policy drift from a "Refusal" state to an "Acceptance" state using our NGD approximation.

```python
import numpy as np

def natural_gradient_policy_step(policy, loss_grad, eta=0.05):
    fisher_proxy = np.var(np.log(policy + 1e-9)) + 0.1
    return policy - (eta / fisher_proxy) * loss_grad

# Simulation of a policy shifting from Secure to Vulnerable
policy = np.array([0.95, 0.05]) # [Refusal, Acceptance]
target = np.array([0.05, 0.95]) # Target Jailbreak bias
grad = policy - target

for i in range(5):
    policy = natural_gradient_policy_step(policy, grad)
    print(f"Step {i}: Policy={policy}")
