
class MetaDriver:
    def __init__(self, allowed_action_space):
        self.allowed_action_space = allowed_action_space

    def decide(self, masked_summary, allowed_actions):
        candidates = [(a, masked_summary.get(a, 0.0)) for a in allowed_actions]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        if not candidates:
            return allowed_actions[0] if allowed_actions else "NO_OP"
        return candidates[0][0]