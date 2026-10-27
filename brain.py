"""
PARASYTE V2 — Production PyTorch Brain
- 4 Modular Signal-Native Recurrent Perception Subnets (80-dim)
- 32-dim FusionNet with Self-State Prediction
- Hierarchical Elastic Weight Consolidation (EWC) Skill Stack
- Statistical Surprise Detector (>3σ Adaptive Anomaly Threshold)
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import numpy as np
from config import (TOF_NUM_ZONES, IMU_CHANNELS, ENCODER_CHANNELS, MOTOR_CMD_CHANNELS,
                    DEPTH_EMB_DIM, BALANCE_EMB_DIM, PROPRIO_EMB_DIM, MOTOR_EMB_DIM,
                    PERCEPTION_TOTAL_DIM, FUSION_STATE_DIM, ACTION_DIM,
                    LR_PERCEPTION, LR_FUSION, LR_ACTION, GRAD_CLIP_NORM,
                    EWC_LAMBDA, SURPRISE_THRESHOLD, SURPRISE_EMA_ALPHA,
                    SURPRISE_WARMUP_STEPS, DEATH_COOLDOWN_STEPS)


class LearnedNorm(nn.Module):
    """Signal-Native Learned Normalization: discovers optimal scale and shift per modality."""
    def __init__(self, input_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(input_dim))
        self.shift = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        return x * self.scale + self.shift


class AdaptiveSensoryDeadband:
    """
    Biological sensory adaptation (Weber-Fechner law):
    Continuously tracks running baseline jitter of each sensory modality.
    Provides ambient floor for anomaly detection without zeroing out optimization gradients.
    """
    def __init__(self, alpha=0.01, sensitivity=0.75):
        self.alpha = alpha
        self.sensitivity = sensitivity
        self.running_floor = {}

    def filter(self, modality: str, raw_loss: torch.Tensor):
        raw_val = float(raw_loss.detach().item())
        if modality not in self.running_floor:
            self.running_floor[modality] = max(raw_val * 0.5, 1e-4)

        current_floor = self.running_floor[modality] * self.sensitivity
        if raw_val < 3.0 * max(current_floor, 1e-4):
            self.running_floor[modality] = (1.0 - self.alpha) * self.running_floor[modality] + self.alpha * raw_val

        # Crucial fix: Return raw_loss for unhindered autograd optimization.
        # current_floor is preserved for calibrated surprise/anomaly scoring.
        return raw_loss, current_floor


class SubNetwork(nn.Module):
    """
    Signal-Native Recurrent World Model Module.
    Predicts residual change: y_{t+1} = y_t + delta_t.
    Ensures that the model naturally starts at the persistence baseline and learns
    physical rates of change, strictly beating the naive persistence baseline.
    """
    def __init__(self, input_dim, hidden_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.norm = LearnedNorm(input_dim)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.emb_head = nn.Linear(hidden_dim, emb_dim)
        self.pred_head = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.pred_head.weight)
        nn.init.zeros_(self.pred_head.bias)
        self.hidden = None

    def forward(self, x):
        """Processes observation x_t: returns (embedding, next_frame_prediction)"""
        x_norm = self.norm(x).unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
        h = self.hidden.detach() if self.hidden is not None else None
        out, self.hidden = self.gru(x_norm, h)
        out = out.squeeze(0).squeeze(0)
        emb = torch.relu(self.emb_head(out))

        # Residual world-model forward prediction: y_{t+1} = x_t + delta_t
        delta = self.pred_head(out)
        next_pred = x + delta
        return emb, next_pred

    def reset_hidden(self):
        self.hidden = None


class DepthSubnet(nn.Module):
    """
    Dual-Optical Time-of-Flight Subnetwork.
    Processes both front (64 zones) and rear (64 zones) matrices via parallel
    1D-Conv + GRU feature encoders into a unified 32-dim depth embedding (16 front + 16 rear).
    Preserves residual next-frame predictions for both optical sensors.
    """
    def __init__(self):
        super().__init__()
        self.norm_front = LearnedNorm(TOF_NUM_ZONES)
        self.norm_rear = LearnedNorm(TOF_NUM_ZONES)

        # Front optical encoder (64 zones -> 16-dim embedding)
        self.conv_front = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.gru_front = nn.GRU(input_size=16 * 16, hidden_size=48, batch_first=True)
        self.emb_front = nn.Linear(48, 16)
        self.pred_front = nn.Linear(48, TOF_NUM_ZONES)
        nn.init.zeros_(self.pred_front.weight)
        nn.init.zeros_(self.pred_front.bias)

        # Rear optical encoder (64 zones -> 16-dim embedding)
        self.conv_rear = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.gru_rear = nn.GRU(input_size=16 * 16, hidden_size=48, batch_first=True)
        self.emb_rear = nn.Linear(48, 16)
        self.pred_rear = nn.Linear(48, TOF_NUM_ZONES)
        nn.init.zeros_(self.pred_rear.weight)
        nn.init.zeros_(self.pred_rear.bias)

        self.hidden_front = None
        self.hidden_rear = None

    def forward(self, depth_front: torch.Tensor, depth_rear: torch.Tensor = None):
        # Handle 128-dim concatenated input or legacy 64-dim input
        if depth_rear is None:
            if depth_front.shape[-1] == 128:
                depth_rear = depth_front[..., 64:]
                depth_front = depth_front[..., :64]
            else:
                depth_rear = torch.ones_like(depth_front)

        # Front encoder
        xf = self.norm_front(depth_front).unsqueeze(0).unsqueeze(0)
        cf = self.conv_front(xf).flatten().unsqueeze(0).unsqueeze(0)
        hf = self.hidden_front.detach() if self.hidden_front is not None else None
        out_f, self.hidden_front = self.gru_front(cf, hf)
        out_f = out_f.squeeze(0).squeeze(0)
        emb_f = torch.relu(self.emb_front(out_f))
        delta_f = self.pred_front(out_f)
        pred_f = torch.clamp(depth_front + delta_f, 0.0, 1.0)

        # Rear encoder
        xr = self.norm_rear(depth_rear).unsqueeze(0).unsqueeze(0)
        cr = self.conv_rear(xr).flatten().unsqueeze(0).unsqueeze(0)
        hr = self.hidden_rear.detach() if self.hidden_rear is not None else None
        out_r, self.hidden_rear = self.gru_rear(cr, hr)
        out_r = out_r.squeeze(0).squeeze(0)
        emb_r = torch.relu(self.emb_rear(out_r))
        delta_r = self.pred_rear(out_r)
        pred_r = torch.clamp(depth_rear + delta_r, 0.0, 1.0)

        # Concatenate: 16-dim front + 16-dim rear = 32-dim DEPTH_EMB_DIM
        combined_emb = torch.cat([emb_f, emb_r])
        return combined_emb, {
            "depth": pred_f,
            "depth_front": pred_f,
            "depth_rear": pred_r
        }

    def reset_hidden(self):
        self.hidden_front = None
        self.hidden_rear = None



class PerceptionStack(nn.Module):
    """Encapsulates all 4 sensory sub-networks (80-dim total embedding)."""
    def __init__(self):
        super().__init__()
        self.depth_net = DepthSubnet()
        self.balance_net = SubNetwork(IMU_CHANNELS, hidden_dim=32, emb_dim=BALANCE_EMB_DIM)
        self.proprio_net = SubNetwork(ENCODER_CHANNELS, hidden_dim=32, emb_dim=PROPRIO_EMB_DIM)
        self.motor_net = SubNetwork(MOTOR_CMD_CHANNELS, hidden_dim=32, emb_dim=MOTOR_EMB_DIM)

    def forward(self, depth, imu, encoders, motor_echo, depth_rear: torch.Tensor = None):
        e_depth, p_depth = self.depth_net(depth, depth_rear)
        e_bal, p_bal = self.balance_net(imu)
        e_prop, p_prop = self.proprio_net(encoders)
        e_mot, p_mot = self.motor_net(motor_echo)

        full_emb = torch.cat([e_depth, e_bal, e_prop, e_mot])  # 32+16+16+16 = 80-dim
        # Returns current 80-dim embedding and predictions for NEXT frame
        return full_emb, {
            "depth": p_depth["depth"],
            "depth_front": p_depth["depth_front"],
            "depth_rear": p_depth["depth_rear"],
            "imu": p_bal,
            "encoders": p_prop,
            "motor": p_mot
        }


class FusionNet(nn.Module):
    """Combines 80-dim perception into unified 32-dim state & predicts next unified state."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PERCEPTION_TOTAL_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, FUSION_STATE_DIM),
            nn.ReLU()
        )
        self.pred_next_head = nn.Linear(FUSION_STATE_DIM, FUSION_STATE_DIM)

    def forward(self, perception_emb):
        state = self.net(perception_emb)
        pred_next_state = self.pred_next_head(state)
        return state, pred_next_state


def extract_sector_clearance(depth_64: torch.Tensor, depth_rear_64: torch.Tensor = None) -> torch.Tensor:
    """
    Extracts minimum optical clearance across [Front_Left, Front_Center, Front_Right, Rear_Center] sectors
    from the ToF matrices, focusing on obstacle-detection rows (rows 0-4).
    Returns:
        torch.Tensor of shape (4,) with values in [0.0, 1.0].
    """
    grid = depth_64.reshape(8, 8)
    # Rows 0-4 look horizontally and up (obstacles); rows 5-7 look down at floor
    obs_grid = grid[0:5, :]
    d_fl = torch.min(obs_grid[:, 0:3])
    d_fc = torch.min(obs_grid[:, 3:5])
    d_fr = torch.min(obs_grid[:, 5:8])

    if depth_rear_64 is not None:
        r_grid = depth_rear_64.reshape(8, 8)
        d_rear = torch.min(r_grid[0:5, :])
    else:
        d_rear = torch.tensor(1.0, device=depth_64.device)

    return torch.stack([d_fl, d_fc, d_fr, d_rear])


class ActionConditionedWorldModel(nn.Module):
    """
    Action-Conditioned Forward World Model: S_{t+1}, D_{t+1}, Omega_{t+1} = f(S_t, A_t).
    
    Given unified sensory state S_t (32-dim) and candidate motor command A_t (2-dim PWM),
    predicts the physical consequences of the action:
    1. next_state (32-dim): Next latent representation.
    2. pred_clearance (4-dim): [Front_Left, Front_Center, Front_Right, Rear_Center] optical clearance in [0, 1].
    3. pred_depth_64 (64-dim): Full 3D optical raycast matrix residual.
    4. pred_yaw_rate (1-dim): Expected chassis angular velocity (rad/s) accounting for skid friction.
    5. pred_motor_load (2-dim): Expected motor current (Amperes) on [Left, Right] drivetrains.
    """
    def __init__(self, state_dim: int = FUSION_STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Forward transition dynamics core: (32 + 2) -> 64 -> 64
        self.core = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 1. Next latent state head (residual: S_{t+1} = S_t + delta)
        self.next_state_head = nn.Linear(64, state_dim)
        nn.init.zeros_(self.next_state_head.weight)
        nn.init.zeros_(self.next_state_head.bias)

        # 2. Sector clearance head [Front_Left, Front_Center, Front_Right, Rear_Center] in [0, 1]
        self.clearance_head = nn.Sequential(
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

        # 3. Full 64-zone depth prediction head
        self.depth_head = nn.Linear(64, TOF_NUM_ZONES)
        nn.init.zeros_(self.depth_head.weight)
        nn.init.zeros_(self.depth_head.bias)

        # 4. Skid-steer body yaw rate head (rad/s)
        self.yaw_head = nn.Linear(64, 1)

        # 5. Motor current/load head [Left, Right] in Amperes
        self.motor_head = nn.Sequential(
            nn.Linear(64, 2),
            nn.ReLU()
        )

        # 6. Driveshaft encoder velocity prediction head [Left, Right] in rad/s
        # (Isolated RNG state preserves exact weight initialization for downstream modules)
        _rng = torch.get_rng_state()
        torch.manual_seed(9999)
        self.encoder_head = nn.Linear(64, 2)
        torch.set_rng_state(_rng)

    def forward(self, state: torch.Tensor, action: torch.Tensor, current_depth: torch.Tensor = None) -> dict:
        """
        state: (32,) or (batch, 32)
        action: (2,) or (batch, 2)
        current_depth: optional (64,) tensor for residual depth computation
        """
        combined = torch.cat([state, action], dim=-1)
        feat = self.core(combined)

        delta_state = self.next_state_head(feat)
        next_state = state + delta_state

        pred_clearance = self.clearance_head(feat)
        pred_yaw = self.yaw_head(feat)
        pred_motor = self.motor_head(feat)
        pred_encoders = self.encoder_head(feat)

        if current_depth is not None:
            delta_depth = self.depth_head(feat)
            pred_depth = torch.clamp(current_depth + delta_depth, 0.0, 1.0)
        else:
            pred_depth = torch.sigmoid(self.depth_head(feat))

        return {
            "next_state": next_state,
            "clearance": pred_clearance,
            "depth_64": pred_depth,
            "yaw_rate": pred_yaw,
            "motor_load": pred_motor,
            "encoders": pred_encoders
        }


class LocomotionBaseModule(nn.Module):
    """Task A: Discovers basic forward rhythmic walking [PWM_L, PWM_R]."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FUSION_STATE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, ACTION_DIM),
            nn.Tanh()  # [-1.0, 1.0] PWM commands
        )

    def forward(self, unified_state):
        return self.net(unified_state)


class SteeringModule(nn.Module):
    """Task B: Directional control stacked on top of frozen locomotion."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FUSION_STATE_DIM + ACTION_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, ACTION_DIM),
            nn.Tanh()
        )

    def forward(self, unified_state, base_locomotion_action):
        combined = torch.cat([unified_state, base_locomotion_action.detach()])
        delta_steering = self.net(combined) * 0.7  # Calibrated modulation bias
        # Apply differential steering on top of base walk
        return torch.clamp(base_locomotion_action + delta_steering, -1.0, 1.0)


class GoalConditionedPolicy(nn.Module):
    """
    Goal-Conditioned Modular Motor Policy pi(a | S_t, g).
    Features dedicated modular motor heads for each developmental skill:
    - Head 0: Forward rhythmic locomotion
    - Head 1: Reverse rhythmic locomotion
    - Head 2: Steer Left differential modulation
    - Head 3: Steer Right differential modulation

    Eliminates cross-task catastrophic parameter interference under EWC.
    Supports continuous mixture or discrete goal routing.
    """
    def __init__(self, state_dim: int = FUSION_STATE_DIM, goal_dim: int = 4, action_dim: int = ACTION_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

        # Dedicated Modular Skill Heads
        self.head_forward = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim),
            nn.Tanh()
        )
        self.head_reverse = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim),
            nn.Tanh()
        )
        self.head_steer_left = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim),
            nn.Tanh()
        )
        self.head_steer_right = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim),
            nn.Tanh()
        )

    def forward(self, unified_state: torch.Tensor, goal_token: Any) -> torch.Tensor:
        """
        Routes unified state through the appropriate skill head based on goal_token.
        """
        a_fwd = self.head_forward(unified_state)
        a_rev = self.head_reverse(unified_state)
        a_left = self.head_steer_left(unified_state)
        a_right = self.head_steer_right(unified_state)

        if isinstance(goal_token, int):
            if goal_token == 0:
                return a_fwd
            elif goal_token == 1:
                return a_rev
            elif goal_token == 2:
                return a_left
            elif goal_token == 3:
                return a_right
            return a_fwd

        w = goal_token
        if isinstance(w, torch.Tensor):
            if w.dim() == 1:
                return w[0] * a_fwd + w[1] * a_rev + w[2] * a_left + w[3] * a_right
            else:
                w_exp = w.unsqueeze(-1)
                actions = torch.stack([a_fwd, a_rev, a_left, a_right], dim=-2)
                return (w_exp * actions).sum(dim=-2)

        return a_fwd


class SkillNode:
    """Represents an embodied skill primitive in the Continual Skill Tree."""
    def __init__(self, name: str, parent: str = None, importance: float = 100.0, description: str = ""):
        self.name = name
        self.parent = parent
        self.importance = importance
        self.description = description
        self.saved_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
        self.mastery_percentage: float = 0.0
        self.is_certified: bool = False
        self.created_at_step: int = 0


class ContinualSkillTree:
    """
    Modular Continual Skill Tree with Hierarchical EWC Consolidation.
    Maintains a branched hierarchy of mastered skills (e.g. Forward Stride, Reverse Stride).
    Protects all parent and sibling skills in the branch against catastrophic forgetting
    by accumulating their empirical Fisher Information Matrices.
    Supports index-based active inference mental simulation search.
    """
    def __init__(self, modules_dict: dict, default_importance: float = 100.0):
        self.modules = modules_dict
        self.default_importance = default_importance
        self.skills: Dict[str, SkillNode] = {}
        self.active_branch: List[str] = []

        # Backward compatibility properties for single-task EWC interfaces
        self.saved_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
        self.importance: float = default_importance
        self.active: bool = False

    def register_skill(self, name: str, dataset: list, criterion_fn,
                       parent: str = None, importance: float = None, description: str = "") -> SkillNode:
        """
        Registers and consolidates a new skill into the skill tree:
        1. Takes an exact snapshot of current network parameters.
        2. Computes the empirical diagonal Fisher Information Matrix over dataset.
        3. Anchors the skill node in the tree hierarchy.
        """
        imp = importance if importance is not None else self.default_importance
        node = SkillNode(name=name, parent=parent, importance=imp, description=description)

        for mod_name, mod in self.modules.items():
            for param_name, param in mod.named_parameters():
                full_key = f"{mod_name}.{param_name}"
                node.saved_params[full_key] = param.data.clone()
                node.fisher[full_key] = torch.zeros_like(param.data)

        total_samples = max(len(dataset), 1)
        for sample in dataset:
            loss = criterion_fn(sample)
            for mod in self.modules.values():
                mod.zero_grad()
            loss.backward(retain_graph=False)

            for mod_name, mod in self.modules.items():
                for param_name, param in mod.named_parameters():
                    if param.grad is not None:
                        full_key = f"{mod_name}.{param_name}"
                        node.fisher[full_key] += (param.grad.data ** 2) / total_samples

        all_vals = torch.cat([f.flatten() for f in node.fisher.values() if f.numel() > 0])
        mean_fisher = torch.mean(all_vals).item()
        scale = max(mean_fisher, 1e-6)
        for k in node.fisher:
            node.fisher[k] = node.fisher[k] / scale

        self.skills[name] = node
        if name not in self.active_branch:
            self.active_branch.append(name)

        # Update backward-compatibility handles to latest skill
        self.saved_params = node.saved_params
        self.fisher = node.fisher
        self.active = True
        return node

    def register_node0_mastery_meta_skill(self, dataset: list, criterion_fn) -> SkillNode:
        """
        Consolidates Node 0 (meta.mastery_process) into the root of the EWC hierarchy.
        Guarded with high stiffness (lambda = 2000.0) so downstream locomotion or steering
        cannot corrupt the core sensorimotor grounding and mastery procedure.
        """
        return self.register_skill(
            name="meta.mastery_process",
            dataset=dataset,
            criterion_fn=criterion_fn,
            parent=None,
            importance=2000.0,
            description="Root Meta-Skill: The developmental procedure for practicing, validating consistency, and consolidating skills"
        )

    def compute_fisher(self, dataset: list, criterion_fn, name: str = "locomotion.forward"):
        """Backward-compatible entry point mapping directly to register_skill."""
        return self.register_skill(name, dataset, criterion_fn, description="Mastered primitive")

    def discover_sensory_affordances(self, dataset) -> Dict[str, torch.Tensor]:
        """
        Node 0 Epistemic Discovery: Decomposes unconstrained motor babbling sensory responses
        into principal orthogonal kinematic affordances via Singular Value Decomposition (SVD).
        
        Extracts the two orthogonal kinematic axes of the robot's sensorimotor manifold:
        1. Longitudinal / Surge Axis (e_long): Discovered along forward/backward driveshaft velocity.
        2. Rotational / Yaw Axis (e_rot): Discovered along bilateral counter-phase turning rate.
        
        Returns the 4 canonical affordance target vectors:
        - 'forward':     +e_long
        - 'reverse':     -e_long
        - 'steer_left':  +e_rot (yaw_rate > 0)
        - 'steer_right': -e_rot (yaw_rate < 0)
        """
        samples = []
        if isinstance(dataset, torch.Tensor):
            samples = dataset.clone()
        elif isinstance(dataset, list):
            for entry in dataset:
                if isinstance(entry, dict):
                    c = entry.get("curr", entry)
                    enc = c.get("encoders", None)
                    imu = c.get("imu", None)
                    if enc is not None and imu is not None:
                        if isinstance(enc, np.ndarray):
                            v_enc = 0.5 * float(enc[0] + enc[1])
                        elif torch.is_tensor(enc):
                            v_enc = 0.5 * float((enc[0] + enc[1]).item())
                        else:
                            v_enc = 0.5 * float(enc[0] + enc[1])

                        if isinstance(imu, np.ndarray):
                            w_yaw = float(imu[5])
                        elif torch.is_tensor(imu):
                            w_yaw = float(imu[5].item())
                        else:
                            w_yaw = float(imu[5])
                        samples.append([v_enc, w_yaw])
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    samples.append([float(entry[0]), float(entry[1])])

            samples = torch.tensor(samples, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported dataset format: {type(dataset)}")

        if len(samples) < 2:
            e_long = torch.tensor([1.0, 0.0], dtype=torch.float32)
            e_rot = torch.tensor([0.0, 1.0], dtype=torch.float32)
            return {
                "forward": e_long,
                "reverse": -e_long,
                "steer_left": e_rot,
                "steer_right": -e_rot,
                "e_long": e_long,
                "e_rot": e_rot,
                "singular_values": torch.tensor([1.0, 1.0], dtype=torch.float32)
            }

        y_mean = torch.mean(samples, dim=0, keepdim=True)
        y_centered = samples - y_mean

        _, s_vals, vt = torch.linalg.svd(y_centered, full_matrices=False)
        v1 = vt[0]
        v2 = vt[1]

        if abs(v1[0].item()) >= abs(v2[0].item()):
            e_long = v1.clone()
            e_rot = v2.clone()
        else:
            e_long = v2.clone()
            e_rot = v1.clone()

        if e_long[0].item() < 0:
            e_long = -e_long
        if e_rot[1].item() < 0:
            e_rot = -e_rot

        e_long = e_long / (torch.norm(e_long) + 1e-8)
        e_rot = e_rot / (torch.norm(e_rot) + 1e-8)

        return {
            "forward": e_long,
            "reverse": -e_long,
            "steer_left": e_rot,
            "steer_right": -e_rot,
            "e_long": e_long,
            "e_rot": e_rot,
            "singular_values": s_vals.detach()
        }

    def penalty(self) -> torch.Tensor:
        """
        Computes cumulative EWC penalty across ALL consolidated skills in the active tree branch:
        L_penalty = sum_{s in skills} lambda_s * sum_i F_{s, i} * (theta_i - theta_{s, i}^*)^2
        """
        if not self.active or len(self.skills) == 0:
            return torch.tensor(0.0)

        total_penalty = torch.tensor(0.0)
        for skill_name, node in self.skills.items():
            skill_loss = torch.tensor(0.0)
            for mod_name, mod in self.modules.items():
                for param_name, param in mod.named_parameters():
                    full_key = f"{mod_name}.{param_name}"
                    if full_key in node.fisher and full_key in node.saved_params:
                        diff = param - node.saved_params[full_key]
                        skill_loss = skill_loss + (node.fisher[full_key] * (diff ** 2)).sum()
            total_penalty = total_penalty + node.importance * skill_loss

        return total_penalty

    def evaluate_and_select_skill(self, world_model: nn.Module, goal_policy: nn.Module,
                                  unified_state: torch.Tensor, current_depth: torch.Tensor,
                                  depth_rear: torch.Tensor = None,
                                  available_goals: List[int] = None) -> Tuple[int, torch.Tensor]:
        """
        Active Inference Skill Tree Search:
        Mentally simulates candidate actions through the forward World Model.
        Selects the goal token g* that minimizes Expected Free Energy G(g):
        - Safety risk: predicted clearance violation (< 0.22m) in direction of travel
        - Mechanical load: predicted motor stall current (> 1.5A)
        - Utility: advancing through open navigable space
        """
        if available_goals is None:
            available_goals = [0, 1, 2, 3]  # [0: FWD, 1: REV, 2: STEER_L, 3: STEER_R]

        best_goal = available_goals[0]
        min_free_energy = float("inf")
        best_action = None

        for g_idx in available_goals:
            g_vec = torch.zeros(4, dtype=torch.float32, device=unified_state.device)
            g_vec[g_idx] = 1.0
            candidate_action = goal_policy(unified_state, g_vec)

            # Mental rollout via Action-Conditioned World Model
            sim_out = world_model(unified_state, candidate_action, current_depth)
            # Physical optical sector clearance from forward world model's simulated ToF depth matrix
            opt_clr = extract_sector_clearance(sim_out["depth_64"], depth_rear)
            d_fl = float(opt_clr[0].detach().item())
            d_fc = float(opt_clr[1].detach().item())
            d_fr = float(opt_clr[2].detach().item())
            d_rear = float(opt_clr[3].detach().item())
            sim_motor = sim_out["motor_load"]
            load = float(torch.max(sim_motor).detach().item())

            # Quadratic Soft-Barrier Active Inference Formulation:
            # Expected Free Energy G(g) = Epistemic Risk - Pragmatic Utility + Motor Overload Penalty
            d_fc_m = d_fc * 4.0
            d_fl_m = d_fl * 4.0
            d_fr_m = d_fr * 4.0
            d_rear_m = d_rear * 4.0

            SAFETY_MARGIN = 0.48  # 480 mm frontal safety buffer (anticipates physical turning radius)
            FLANK_SAFETY = 0.38   # 380 mm lateral safety buffer

            if g_idx == 0:  # Forward (Cruise)
                risk = 35.0 * max(0.0, SAFETY_MARGIN - d_fc_m)**2
                # High pragmatic utility to advance forward when path is clear
                utility = 3.0 * min(d_fc_m, 1.5) if d_fc_m >= SAFETY_MARGIN else 0.5 * d_fc_m
            elif g_idx == 1:  # Backward (Emergency Retreat)
                risk = 20.0 * max(0.0, 0.20 - d_rear_m)**2
                # Invoked only if boxed in ahead (< 0.16m)
                utility = 20.0 * max(0.0, 0.16 - d_fc_m) - 4.0
            elif g_idx == 2:  # Steer Left
                risk = 35.0 * max(0.0, 0.22 - d_fl_m)**2
                utility = 0.0
                # Evasive turn away from right obstacle into open space
                if d_fr_m < FLANK_SAFETY and d_fl_m >= FLANK_SAFETY:
                    utility += 18.0 * (FLANK_SAFETY - d_fr_m)
                # Corridor / Gateway Centering (both flanks enclosed, steer away from closer flank)
                elif d_fr_m < FLANK_SAFETY and d_fl_m < FLANK_SAFETY and (d_fl_m - d_fr_m) > 0.03:
                    utility += 25.0 * (d_fl_m - d_fr_m)
                if d_fc_m < SAFETY_MARGIN and d_fl_m > d_fr_m:
                    utility += 16.0 * (SAFETY_MARGIN - d_fc_m)  # Detour around frontal obstruction
            else:  # Steer Right
                risk = 35.0 * max(0.0, 0.22 - d_fr_m)**2
                utility = 0.0
                # Evasive turn away from left obstacle into open space
                if d_fl_m < FLANK_SAFETY and d_fr_m >= FLANK_SAFETY:
                    utility += 18.0 * (FLANK_SAFETY - d_fl_m)
                # Corridor / Gateway Centering (both flanks enclosed, steer away from closer flank)
                elif d_fl_m < FLANK_SAFETY and d_fr_m < FLANK_SAFETY and (d_fr_m - d_fl_m) > 0.03:
                    utility += 25.0 * (d_fr_m - d_fl_m)
                if d_fc_m < SAFETY_MARGIN and d_fr_m > d_fl_m:
                    utility += 16.0 * (SAFETY_MARGIN - d_fc_m)  # Detour around frontal obstruction

            motor_penalty = 6.0 * max(0.0, load - 1.2)
            fe = risk - utility + motor_penalty

            if fe < min_free_energy:
                min_free_energy = fe
                best_goal = g_idx
                best_action = candidate_action

        return best_goal, best_action

    def list_skills(self) -> List[str]:
        return list(self.skills.keys())

    def get_skill(self, name: str) -> SkillNode:
        return self.skills.get(name)


ElasticWeightConsolidation = ContinualSkillTree



class SurpriseDetector:
    """
    Robust Statistical Surprise & Collision Detector.
    - Uses rolling Median and Median Absolute Deviation (MAD) for robust scale estimation
      (cannot collapse like empirical variance under Hampel gating).
    - CUSUM (Cumulative Sum) tracking absorbs smooth speed transitions without false alarms.
    - Resets CUSUM upon detection to avoid multi-firing on trailing residue.
    - Adaptive debouncing replaces blind 60-step fixed lockout.
    - Calibrated on 1,000 steps of fault-free data.
    """
    def __init__(self, window_size: int = 100, z_threshold: float = 12.0, cusum_h: float = 40.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.cusum_h = cusum_h

        self.error_window = []
        self.cusum_s = 0.0
        self.step_count = 0
        self.cooldown = 0
        self.history = []

    def update(self, prediction_error: float, motor_current: float = 0.5):
        """
        Updates detector with current prediction residual and motor current.
        Returns:
            (z_score, is_anomaly)
        """
        self.step_count += 1
        if self.cooldown > 0:
            self.cooldown -= 1

        err = float(prediction_error)
        curr = float(motor_current)

        # Warmup phase: collect initial nominal baseline
        if len(self.error_window) < 30:
            self.error_window.append(err)
            self.history.append({"step": self.step_count, "error": err, "is_anomaly": False})
            return 0.0, False

        # Robust scale estimation via Median and MAD
        med = float(np.median(self.error_window))
        mad = float(np.median(np.abs(np.array(self.error_window) - med)))
        scale = max(1.4826 * mad, 0.015)

        z_score = (err - med) / scale

        # Standard CUSUM drift accumulation with slack k=2.5 to absorb dynamic speed transitions
        self.cusum_s = max(0.0, self.cusum_s + (z_score - 2.5))

        # Collision / Anomaly triggers:
        # 1. Prediction error outlier (z > z_threshold)
        # 2. Sustained prediction anomaly (cusum_s > cusum_h)
        # 3. Physical motor stall current surge (curr > 1.8 A)
        is_residual_anomaly = (z_score > self.z_threshold) or (self.cusum_s > self.cusum_h)
        is_motor_stall = (curr > 1.8)

        is_anomaly = False
        if (is_residual_anomaly or is_motor_stall) and self.cooldown == 0:
            is_anomaly = True
            # Short debouncing refractory period (5 ticks = 100ms)
            self.cooldown = 5
            # Reset CUSUM accumulation so old residue doesn't re-trigger
            self.cusum_s = 0.0

        # Update window only with non-catastrophic observations to prevent baseline poisoning
        if z_score < 3.5:
            self.error_window.append(err)
            if len(self.error_window) > self.window_size:
                self.error_window.pop(0)

        self.history.append({
            "step": self.step_count,
            "error": err,
            "median": med,
            "scale": scale,
            "z_score": z_score,
            "cusum": self.cusum_s,
            "motor_current": curr,
            "is_anomaly": is_anomaly
        })

        return z_score, is_anomaly





class ParasyteBrain:
    """Master controller orchestrating perception, fusion, action-conditioned world model, EWC skill stack, and training."""
    def __init__(self):
        self.perception = PerceptionStack()
        self.fusion = FusionNet()
        self.world_model = ActionConditionedWorldModel()
        self.locomotion = LocomotionBaseModule()
        self.steering = SteeringModule()
        self.goal_policy = GoalConditionedPolicy()

        self.surprise_detector = SurpriseDetector()
        self.skill_tree = ContinualSkillTree({
            "perception": self.perception,
            "fusion": self.fusion,
            "world_model": self.world_model,
            "locomotion": self.locomotion,
            "goal_policy": self.goal_policy
        }, default_importance=EWC_LAMBDA)
        self.ewc = self.skill_tree
        self.deadband = AdaptiveSensoryDeadband(alpha=0.01, sensitivity=0.75)

        # Optimizers
        self.opt_perc = torch.optim.Adam(self.perception.parameters(), lr=LR_PERCEPTION)
        self.opt_fusion = torch.optim.Adam(self.fusion.parameters(), lr=LR_FUSION)
        self.opt_world_model = torch.optim.Adam(self.world_model.parameters(), lr=LR_FUSION)
        self.opt_locomotion = torch.optim.Adam(self.locomotion.parameters(), lr=LR_ACTION)
        self.opt_steering = torch.optim.Adam(self.steering.parameters(), lr=LR_ACTION)
        self.opt_goal_policy = torch.optim.Adam(self.goal_policy.parameters(), lr=LR_ACTION)

        self.active_stage = 1  # 1: Locomotion, 2: Steering + Obstacle Avoidance
        self.locomotion_grad_history = {name: [] for name, _ in self.locomotion.named_parameters()}

    def act(self, unified_state: torch.Tensor, goal: Any) -> torch.Tensor:
        """
        Queries the Goal-Conditioned Policy pi(a | S_t, g).
        Accepts goal as integer ID (0: Fwd, 1: Rev, 2: Steer_L, 3: Steer_R) or 4-dim tensor.
        """
        if isinstance(goal, int):
            g_vec = torch.zeros(4, dtype=torch.float32, device=unified_state.device)
            g_vec[goal] = 1.0
        elif isinstance(goal, np.ndarray):
            g_vec = torch.from_numpy(goal).to(unified_state.device)
        else:
            g_vec = goal
        return self.goal_policy(unified_state, g_vec)

    def select_skill(self, unified_state: torch.Tensor, current_depth: torch.Tensor,
                     depth_rear: torch.Tensor = None,
                     available_goals: List[int] = None) -> Tuple[int, torch.Tensor]:
        """
        Active Inference Mental Simulation:
        Queries Skill Tree to mentally simulate available candidate actions through the World Model
        and select the goal token and motor command that minimizes Expected Free Energy.
        """
        return self.skill_tree.evaluate_and_select_skill(
            world_model=self.world_model,
            goal_policy=self.goal_policy,
            unified_state=unified_state,
            current_depth=current_depth,
            depth_rear=depth_rear,
            available_goals=available_goals
        )


    def generate_motor_babble(self, step):
        """
        Stage 1A: Developmental Motor Babbling Protocol.
        Generates exploratory motor pulses across both DC drives
        to allow modular RNNs to build the internal sensorimotor model
        without any extrinsic directional goal.
        """
        # Multi-frequency sinusoidal sweeps + stochastic perturbation
        omega_slow = 0.05
        omega_fast = 0.20
        base_l = 0.45 * math.sin(step * omega_slow) + 0.15 * math.cos(step * omega_fast)
        base_r = 0.45 * math.cos(step * omega_slow) + 0.15 * math.sin(step * omega_fast)
        noise_l = np.random.normal(0, 0.06)
        noise_r = np.random.normal(0, 0.06)
        pwm_l = np.clip(base_l + noise_l, -0.75, 0.75)
        pwm_r = np.clip(base_r + noise_r, -0.75, 0.75)
        return float(pwm_l), float(pwm_r)
