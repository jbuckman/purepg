class BaseEnv:
    game_type = None
    state_shape = None
    action_count = None
    tokens = None
    state_continuous = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)

    def render(self):
        raise Exception('Unimplemented')

    def serialize(self):
        raise Exception('Unimplemented')

    @classmethod
    def deserialize(self, serial_env):
        raise Exception('Unimplemented')

    def state_rep(self, slot=None):
        raise Exception('Unimplemented')

    @property
    def legal_actions(self):
        return None

    @property
    def has_action_restrictions(self):
        return self.legal_actions != None

    def step(self, action):
        raise Exception('Unimplemented')