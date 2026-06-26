



class Wow:
    # Experiment
    device: str = "gpu"  # one of: cpu, gpu, tpu. JAX selects the matching backend when available.
    env: str = "halfcheetah-medium-expert-v2"
    seed: int = 0

    def __init__(self, device = "gpu"):
        self.device = device


    def seminlee(self):
        return self.device + self.env


wow = Wow()

print(wow.device)
print(wow.env)

print(wow.seminlee())


wow2 = Wow(
    device = "cpu"
)

print(wow2.device)

print(wow2.seminlee())