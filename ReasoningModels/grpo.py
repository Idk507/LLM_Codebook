import torch
from torch.nn import functional as F

# Assume: 
# - pi_theta: the current policy (a language model)
# - pi_anchor: the frozen anchor policy (SFT model)
# - reward_model: a model that outputs scalar reward
# - tokenizer: tokenizer for the LLM
# - beta: KL penalty coefficient

def compute_kl_divergence(logits_theta, logits_anchor):
    # Compute KL(pi_theta || pi_anchor)
    probs_theta = F.softmax(logits_theta, dim=-1)
    log_probs_theta = F.log_softmax(logits_theta, dim=-1)
    log_probs_anchor = F.log_softmax(logits_anchor, dim=-1)
    kl = torch.sum(probs_theta * (log_probs_theta - log_probs_anchor), dim=-1)
    return kl

def compute_grpo_loss(inputs, pi_theta, pi_anchor, reward_model, beta):
    # Forward pass
    logits_theta = pi_theta(**inputs).logits
    logits_anchor = pi_anchor(**inputs).logits.detach()

    # Get token-level log probs
    log_probs_theta = F.log_softmax(logits_theta, dim=-1)
    
    # Sample sequences
    sampled_tokens = torch.argmax(logits_theta, dim=-1)

    # Compute reward
    texts = tokenizer.batch_decode(sampled_tokens, skip_special_tokens=True)
    rewards = torch.tensor([reward_model(t) for t in texts]).to(logits_theta.device)

    # Compute KL
    kl = compute_kl_divergence(logits_theta, logits_anchor)

    # Final loss
    total_reward = rewards - beta * kl.mean(dim=-1)
    loss = -torch.mean(total_reward)

    return loss
def dpo_loss(pi_theta, pi_ref, prompt, chosen, rejected, beta):
    log_probs_chosen = pi_theta.log_prob(prompt, chosen)
    log_probs_rejected = pi_theta.log_prob(prompt, rejected)
    
    ref_log_probs_chosen = pi_ref.log_prob(prompt, chosen).detach()
    ref_log_probs_rejected = pi_ref.log_prob(prompt, rejected).detach()
    
    # Compute log-ratio
    logratios = (log_probs_chosen - log_probs_rejected) - beta * (ref_log_probs_chosen - ref_log_probs_rejected)
    loss = -F.logsigmoid(logratios).mean()
    return loss
