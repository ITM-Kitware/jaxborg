"""Reusable recording wrapper for CybORG's np_random objects.

Captures all random calls (choice, integers, random) while delegating
to the original RNG, so tests can inspect what CybORG drew.
"""


class RecordingNPRandom:
    """Proxy wrapper that records all random calls while delegating to the original."""

    def __init__(self, orig):
        self._orig = orig
        self.log = []

    def choice(self, a, *args, **kwargs):
        result = self._orig.choice(a, *args, **kwargs)
        if hasattr(a, "__len__"):
            n = len(a)
            items = list(a)
            try:
                idx = items.index(result)
            except ValueError:
                idx = 0
        else:
            n = int(a)
            idx = int(result)
        self.log.append(("choice", idx, n, result))
        return result

    def integers(self, high, *args, **kwargs):
        result = self._orig.integers(high, *args, **kwargs)
        if len(args) >= 1:
            low = int(high)
            high_val = int(args[0])
        else:
            low = 0
            high_val = int(high)
        self.log.append(("integers", int(result), low, high_val))
        return result

    def random(self, *args, **kwargs):
        result = self._orig.random(*args, **kwargs)
        self.log.append(("random", float(result), None))
        return result

    def shuffle(self, *args, **kwargs):
        return self._orig.shuffle(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._orig, name)
