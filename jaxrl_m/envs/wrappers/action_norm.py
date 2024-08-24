import gym


class UnnormalizeAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env, action_metadata: dict):
        self.action_mean = action_metadata["mean"]
        self.action_std = action_metadata["std"]
        super().__init__(env)

    def action(self, action):
        """
        Un-normalizes the action
        """
        action = (action * self.action_std) + self.action_mean
        return action
