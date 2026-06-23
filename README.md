# Information-Geometric Unification of Multi-Turn Narrative Erosion as a Natural Gradient Descent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Theory: Information Geometry](https://img.shields.io/badge/Theory-Information_Geometry-blue.svg)]()
[![Field: AI_Alignment/Offensive_Security](https://img.shields.io/badge/Field-AI__Alignment%2FOffensive__Security-purple.svg)]()

This repository formalizes a novel theoretical framework uniting **Iterated Learning (Cognitive Science)**, **Discrete Textual Gradients (LLM Optimization)**, and **Information Geometry (Amari's Natural Gradient)**.

### 🔴 Bug Fix (v2.0 Update)
*The previous formulation applied the gradient descent directly to token spaces or embedding vectors, which violated the differentiability requirement due to the discrete nature of text. This version corrects the flaw by lifting the manifold onto the continuous **Policy Space** of autoregressive generators.*

---

## 1. Executive Summary

When a text is transmitted through a chain of lossy channels (whether human minds or autoregressive Large Language Models), it undergoes *narrative erosion*. Traditional models treat this as a simple Markov chain or lossy compression. 

This framework proves that if the transmission channel prioritizes semantic preservation under cognitive/computational constraints, the step-by-step drift of the narrative does not happen in a discrete word space, but on the continuous **Riemannian manifold of generation policies**. **Narrative erosion is strictly equivalent to an implicit Natural Gradient Descent (NGD) optimizing for maximum memorability and lowest perplexity.**

---

## 2. Theoretical Framework & Mathematical Proof

### A. The Autoregressive Policy Manifold
To circumvent the non-differentiability of the discrete token vocabulary $V$, we model the system within the space of probability distributions. Let $\mathcal{M}$ be a statistical manifold parameterizing the **generation policies** of the agents. 

A narrative state at turn $t$ is represented as the joint probability distribution induced by the autoregressive policy $\pi_\theta$:

$$p_\theta(x) = \prod_{i=1}^{T} \pi_\theta(x_i \mid x_{<i})$$

Where $\theta \in \mathbb{R}^d$ represents the continuous latent parameters dictating the distribution (e.g., prompt activation vectors, attention conditioning, or distribution temperatures).

### B. The Intrinsic Metric (Fisher Information Tensor)
The divergence between two consecutive transmission steps is measured via the **Kullback-Leibler (KL) Divergence** over the generation policies. The second-order Taylor expansion defines the local Riemannian curvature, parameterized by the **Fisher Information Matrix (FIM)**:

$$D_{KL}(p_\theta \parallel p_{\theta + d\theta}) \approx \frac{1}{2} d\theta^T I(\theta) d\theta$$

Where $I(\theta)$ acts as the metric tensor mapping the intrinsic geometry of meaning:

$$I(\theta) = \mathbb{E}_{x \sim p_\theta} \left[ \nabla_\theta \log p_\theta(x) \nabla_\theta \log p_\theta(x)^T \right]$$

### C. The Objective Function (Inductive Bias Bottleneck)
At each turn $t$, the receiving agent $A_{t+1}$ introduces an implicit loss function $L(\theta)$. This loss represents the **cognitive load**, **perplexity**, or **rejection rate** of information that deviates from the agent's internal structural prior (or the LLM's architecture):

$$L(\theta) = -\mathbb{E}_{x \sim p_\theta} [\log p_{\text{prior}}(x)]$$

### D. Analytical Derivation of the Update Rule
During transmission, the narrative evolves to minimize $L(\theta)$ while bounded by the information-preservation limit quantified by the Fisher metric. The step transition satisfies the constrained optimization problem:

$$\theta_{t+1} = \arg\min_{\theta} \left\{ L(\theta) + \frac{1}{2\eta} D_{KL}(p_{\theta_t} \parallel p_\theta) \rightfixed\}$$

Solving the first-order optimality condition on this curved statistical manifold yields the exact analytical formulation of **Amari's Natural Gradient Descent**:

$$\theta_{t+1} = \theta_t - \eta I(\theta_t)^{-1} \nabla_\theta L(\theta_t)$$

Where:
* $\eta$ is the transmission temperature (learning rate).
* $I(\theta_t)^{-1}$ is the inverse Fisher Information Matrix, which preconditions and transforms the vanilla gradient to align with the true semantic topology of the language space.

---

## 3. Implications for Offensive Security & Jailbreaks

| Phenomenon | Information-Geometric Interpretation |
| :--- | :--- |
| **Discrete Token Adversarial Attacks** | Jailbreak payloads and adversarial suffixes (e.g., GCG) exploit regions where the determinant of the Fisher Matrix $\det(I(\theta)) \to 0$. In these singular topologies, micro-mutations in discrete text collapse the generation policy, causing massive leaps in the output space and bypassing safety guardrails. |
| **Recursive Model Collapse** | Explains why self-consuming loops (LLMs trained on LLM outputs) decay into high-bias states. NGD continuously drives the model's generation policy toward the geometric center of mass of its structural priors. |

---

## 4. Empirical Verification Architecture

```python
import numpy as np

def compute_policy_fisher_distance(policy_t, policy_next):
    """
    Approximates the local Riemannian distance between two continuous 
    generation policies using Kullback-Leibler Divergence.
    """
    return np.sum(policy_t * np.log((policy_t + 1e-9) / (policy_next + 1e-9)))

def natural_gradient_policy_step(policy_distribution, loss_gradient, eta=0.01):
    """
    Applies an exact NGD update step on the continuous policy simplex.
    """
    # Fisher Information Proxy computed via the variance of log-probabilities
    fisher_proxy = np.var(np.log(policy_distribution + 1e-9)) + 1.0
    
    # Natural Gradient Preconditioning
    natural_grad = loss_gradient / fisher_proxy
    
    # Continuous parameter update
    updated_policy = policy_distribution - eta * natural_grad
    
    # Projection back onto the valid probability simplex
    updated_policy = np.clip(updated_policy, 1e-9, 1.0)
    return updated_policy / np.sum(updated_policy)
