"""Countdown domain backbone: canonicalizer Phi, executor, goal test, exact oracle.

Countdown / 24-game semantics
-----------------------------
A *state* is a multiset of currently-available exact values plus a fixed target.
An operation combines two distinct available values a, b into one result a o b
(consuming both, producing one), repeated until a single value remains.
Success (goal) is reached at a single-value state whose value equals the target.

Canonicalization (the Phi merge)
--------------------------------
Two states are equivalent iff their value multisets and target agree. We canonicalize
by sorting the values tuple deterministically (Fractions are totally ordered). `canon`
returns the hashable merge key; `apply` always returns states with sorted values so
that distinct operation orders that reach the same multiset collapse to one node.

Operation set and the DIVISION POLICY
-------------------------------------
Operations are taken over the CANONICAL (sorted) values. For indices i < j with
a = values[i] <= b = values[j]:

  '+'  -> a + b
  '*'  -> a * b
  '-'  -> b - a            (larger minus smaller; non-negative because order is canonical)
  '/'  -> b / a            (larger divided by smaller)

DIVISION POLICY (chosen and fixed):
    '/' is legal whenever the divisor is non-zero. Because we work entirely in
    `fractions.Fraction`, division is ALWAYS exact: the result is the exact rational
    b / a, which need NOT be an integer. We therefore do NOT require an integer
    quotient. The only restriction is divisor != 0 (here the divisor is the smaller
    value a, so '/' is disallowed exactly when the smallest of the pair is 0).

    Example: [7, 2] -> canonical (2, 7) -> '/' yields Fraction(7, 2), a legal op.

No floats anywhere; every numeric value is a `fractions.Fraction` for exactness.

Operation de-duplication
------------------------
When a multiset contains repeated values (e.g. (2, 2, 3)), different index pairs can
produce identical resulting multisets. `legal_ops` de-duplicates by the canonical key
of the resulting state, keeping one representative operation per distinct outcome.
"""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class State:
    values: tuple  # tuple[Fraction, ...], not required to be pre-sorted
    target: Fraction


# Op = (i, j, symbol), i < j indices into the CANONICAL (sorted) values tuple,
# symbol in {'+', '-', '*', '/'}.
Op = tuple


def canon(state):
    """Return the Phi merge key: (sorted values tuple, target).

    Sorting is deterministic because Fractions are totally ordered.
    """
    return (tuple(sorted(state.values)), state.target)


def _combine(a, b, symbol):
    """Compute a o b where a <= b (canonical order). Returns a Fraction.

    '-' is larger minus smaller (b - a); '/' is larger over smaller (b / a).
    Caller guarantees legality (e.g. non-zero divisor for '/').
    """
    if symbol == '+':
        return a + b
    if symbol == '*':
        return a * b
    if symbol == '-':
        return b - a
    if symbol == '/':
        return b / a
    raise ValueError(f"unknown operation symbol: {symbol!r}")


def legal_ops(state):
    """All legal operations over the canonicalized values, de-duplicated by outcome.

    Indices i < j refer to the SORTED values tuple. For each pair we emit every
    applicable symbol; '/' is emitted whenever the divisor (the smaller value) is
    non-zero (see DIVISION POLICY in the module docstring). Operations producing an
    identical resulting multiset are collapsed to a single representative.
    """
    vals = tuple(sorted(state.values))
    n = len(vals)
    ops = []
    seen_results = set()
    for i in range(n):
        for j in range(i + 1, n):
            a = vals[i]  # smaller (canonical order)
            b = vals[j]  # larger
            for symbol in ('+', '-', '*', '/'):
                if symbol == '/' and a == 0:
                    # divisor is the smaller value a; division by zero is illegal.
                    continue
                op = (i, j, symbol)
                result_state = apply(state, op)
                key = canon(result_state)
                if key in seen_results:
                    continue
                seen_results.add(key)
                ops.append(op)
    return ops


def apply(state, op):
    """Apply `op` to `state`, returning a NEW State with canonicalized (sorted) values.

    `op` indices refer to the SORTED values tuple. The two consumed values are
    replaced by their single combined result; the returned State's values are sorted.
    """
    i, j, symbol = op
    vals = tuple(sorted(state.values))
    a = vals[i]
    b = vals[j]
    result = _combine(a, b, symbol)
    remaining = [v for k, v in enumerate(vals) if k != i and k != j]
    remaining.append(result)
    return State(values=tuple(sorted(remaining)), target=state.target)


def is_goal(state):
    """True iff exactly one value remains and it equals the target."""
    return len(state.values) == 1 and state.values[0] == state.target


def _reachable_memo(state, memo):
    key = canon(state)
    if key in memo:
        return memo[key]
    if is_goal(state):
        memo[key] = True
        return True
    if len(state.values) == 1:
        # single value but not equal to target -> dead end.
        memo[key] = False
        return False
    # Guard against revisiting during the current branch's recursion is unnecessary:
    # each op strictly reduces the value count, so the search is a finite DAG.
    result = False
    for op in legal_ops(state):
        if _reachable_memo(apply(state, op), memo):
            result = True
            break
    memo[key] = result
    return result


def reachable(state):
    """Exact, memoized backward oracle: True iff some legal op sequence reaches a goal.

    Memoized on `canon(state)` so equivalent multisets are explored once.
    """
    return _reachable_memo(state, {})


def _solve_one_memo(state, memo):
    """Return a witness path (list of ops) to a goal, or None. Memoizes solvability
    only for known-unsolvable keys to prune; solvable paths are returned directly.
    """
    key = canon(state)
    if is_goal(state):
        return []
    if len(state.values) == 1:
        return None
    if memo.get(key) is False:
        return None
    for op in legal_ops(state):
        sub = _solve_one_memo(apply(state, op), memo)
        if sub is not None:
            return [op] + sub
    memo[key] = False
    return None


def solve_one(state):
    """Return one witness solution path (list of Ops) to a goal, or None."""
    return _solve_one_memo(state, {})


def parse_target_problem(numbers, target):
    """Build a State from a list of ints/numbers and a target, all as Fractions."""
    values = tuple(Fraction(n) for n in numbers)
    return State(values=values, target=Fraction(target))


def render(state):
    """Minimal, stable rendering used later as a prompt.

    Values are rendered in canonical (sorted) order so equivalent states render
    identically.
    """
    vals = tuple(sorted(state.values))
    nums = ", ".join(_fmt(v) for v in vals)
    return f"numbers: {nums} | target: {_fmt(state.target)} | propose ONE next operation"


def _fmt(v):
    """Format a Fraction as an integer when whole, else as 'p/q'. Exact, no floats."""
    if v.denominator == 1:
        return str(v.numerator)
    return f"{v.numerator}/{v.denominator}"
