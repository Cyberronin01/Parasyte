"""
PARASYTE Mastery Evaluator & Certification Referee
===================================================
Tracks empirical trial performance, computes true 0-100% mastery scores,
and certifies skills for EWC consolidation and child unlocking.
"""

from typing import Dict, List, Optional, Tuple, Any
from .skill_contract import SkillContract, TrialObservation


class SkillStatus:
    LOCKED = "LOCKED"
    LEARNING = "LEARNING"
    PRACTICING = "PRACTICING"
    MASTERED = "MASTERED"


class SkillProgress:
    """Tracks progress, trial history, and certification status for a single skill."""
    def __init__(self, contract: SkillContract, prerequisites: List[str] = None):
        self.contract = contract
        self.prerequisites = prerequisites or []
        self.status = SkillStatus.LOCKED if self.prerequisites else SkillStatus.LEARNING
        self.trial_history: List[Dict[str, Any]] = []
        self.mastery_pct: float = 0.0
        self.consecutive_passes: int = 0
        self.is_certified: bool = False

    def record_trial(self, passed: bool, score: float, diagnostics: Dict[str, Any]):
        self.trial_history.append({
            "trial_id": len(self.trial_history) + 1,
            "passed": passed,
            "score": score,
            "diagnostics": diagnostics
        })

        if passed:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = max(0, self.consecutive_passes - 1)

        # Compute rolling mastery over last 5 trials
        window = self.trial_history[-5:]
        avg_score = sum(t["score"] for t in window) / len(window)
        pass_ratio = sum(1 for t in window if t["passed"]) / len(window)

        # Mastery is joint function of consistency and quality
        self.mastery_pct = round(float(avg_score * pass_ratio * 100.0), 1)

        # 100.0% Certification Gauntlet:
        # Requires 5/5 consecutive passes with perfect 100% score (pass_ratio == 1.0 and avg_score == 1.0)
        if self.mastery_pct >= 99.9 and self.consecutive_passes >= 5:
            self.status = SkillStatus.MASTERED
            self.is_certified = True
        elif self.mastery_pct >= 60.0:
            self.status = SkillStatus.PRACTICING
        else:
            self.status = SkillStatus.LEARNING


class MasteryEvaluator:
    """
    Master Evaluator managing the developmental progress of the entire skill tree.
    """
    def __init__(self):
        self.skills: Dict[str, SkillProgress] = {}

    def register_skill(self, contract: SkillContract, prerequisites: List[str] = None) -> SkillProgress:
        progress = SkillProgress(contract, prerequisites)
        self.skills[contract.name] = progress
        self._update_unlocks()
        return progress

    def evaluate_trial(self, skill_name: str, trial: TrialObservation) -> Tuple[bool, float, float, str]:
        """
        Evaluates a trial for the given skill, updates mastery, and returns:
        (passed, score, mastery_pct, status)
        """
        if skill_name not in self.skills:
            raise ValueError(f"Skill '{skill_name}' not registered in MasteryEvaluator!")

        progress = self.skills[skill_name]
        passed, score, diag = progress.contract.evaluate_trial(trial)
        progress.record_trial(passed, score, diag)

        self._update_unlocks()
        return passed, score, progress.mastery_pct, progress.status

    def _update_unlocks(self):
        """Unlocks locked skills whose prerequisites are fully certified/mastered."""
        for name, progress in self.skills.items():
            if progress.status == SkillStatus.LOCKED:
                all_prereqs_mastered = all(
                    prereq in self.skills and self.skills[prereq].is_certified
                    for prereq in progress.prerequisites
                )
                if all_prereqs_mastered:
                    progress.status = SkillStatus.LEARNING

    def get_skill_tree_summary(self) -> List[Dict[str, Any]]:
        """Returns structured summary for UI dashboard and logging."""
        summary = []
        for name, p in self.skills.items():
            summary.append({
                "skill": name,
                "goal_id": p.contract.goal_id,
                "status": p.status,
                "mastery_pct": p.mastery_pct,
                "certified": p.is_certified,
                "trials": len(p.trial_history),
                "prerequisites": p.prerequisites
            })
        return summary
