from __future__ import annotations
from fractions import Fraction as F
from dataclasses import dataclass
import itertools, math, json
from pathlib import Path
BITS = 384
SCALE = 1 << BITS

def ceildiv(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError('positive denominator required')
    return -(-a // b)

@dataclass(frozen=True)
class I:
    lo: int
    hi: int

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError('reversed interval')

    @classmethod
    def of(cls, x):
        if isinstance(x, cls):
            return x
        q = F(x)
        return cls(q.numerator * SCALE // q.denominator, ceildiv(q.numerator * SCALE, q.denominator))

    @property
    def lower(self):
        return F(self.lo, SCALE)

    @property
    def upper(self):
        return F(self.hi, SCALE)

    @property
    def width(self):
        return F(self.hi - self.lo, SCALE)

    def __add__(self, other):
        b = I.of(other)
        return I(self.lo + b.lo, self.hi + b.hi)
    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -I.of(other)

    def __rsub__(self, other):
        return I.of(other) + -self

    def __mul__(self, other):
        b = I.of(other)
        v = [self.lo * b.lo, self.lo * b.hi, self.hi * b.lo, self.hi * b.hi]
        return I(min(v) // SCALE, ceildiv(max(v), SCALE))
    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError('interval contains zero')
        if self.hi < 0:
            return -(-self).reciprocal()
        return I(SCALE * SCALE // self.hi, ceildiv(SCALE * SCALE, self.lo))

    def __truediv__(self, other):
        return self * I.of(other).reciprocal()

    def __rtruediv__(self, other):
        return I.of(other) * self.reciprocal()

    def __pow__(self, n: int):
        if not isinstance(n, int) or n < 0:
            raise ValueError('nonnegative integer power required')
        ans = I.of(1)
        b = self
        while n:
            if n & 1:
                ans = ans * b
            b = b * b
            n //= 2
        return ans

    def contains(self, x):
        return self.lower <= F(x) <= self.upper

    def dump(self, places=60):
        k = 10 ** places
        lo = self.lo * k // SCALE
        hi = ceildiv(self.hi * k, SCALE)

        def fmt(v):
            sign = '-' if v < 0 else ''
            v = abs(v)
            return sign + str(v // k) + '.' + str(v % k).zfill(places)
        return {'lower': fmt(lo), 'upper': fmt(hi), 'width_upper': math.nextafter(float(self.width), math.inf) if self.width else 0.0, 'dyadic_lower_numerator': str(self.lo), 'dyadic_upper_numerator': str(self.hi), 'denominator_power_of_two': BITS}

def exp_negative(x: F, degree=96) -> I:
    x = F(x)
    if x < 0:
        raise ValueError('x must be nonnegative')
    if x == 0:
        return I.of(1)
    if degree < 1:
        raise ValueError('degree must be positive')
    k = 0
    r = x
    while r > F(1, 2):
        r /= 2
        k += 1
    rr = I.of(r)
    s = I.of(1)
    term = I.of(1)
    for n in range(1, degree + 1):
        term = term * rr / n
        s = s + term
    tail = F(degree + 2, (degree + 1) * math.factorial(degree + 1))
    eb = I(s.lo, (s + tail).hi).reciprocal()
    for _ in range(k):
        eb = eb * eb
    return eb

def eighth_root_unit(r: F) -> I:
    r = F(r)
    if not 0 < r < 1:
        raise ValueError('0<r<1 required')
    lo = 0
    hi = SCALE
    rhs = r.numerator * SCALE ** 8
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid ** 8 * r.denominator <= rhs:
            lo = mid
        else:
            hi = mid
    return I(lo, hi)

def determinant(A):
    n = len(A)
    if not n or any((len(r) != n for r in A)):
        raise ValueError('nonempty square matrix required')
    out = 0
    for perm in itertools.permutations(range(n)):
        sign = (-1) ** sum((perm[i] > perm[j] for i in range(n) for j in range(i + 1, n)))
        term = sign
        for i, j in enumerate(perm):
            term = term * A[i][j]
        out = out + term
    return out

def solve(A, b):
    n = len(A)
    if len(b) != n or any((len(r) != n for r in A)):
        raise ValueError('square system required')
    a = [[F(x) for x in row] + [F(y)] for row, y in zip(A, b)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col]), None)
        if pivot is None:
            raise ArithmeticError('singular exact system')
        a[col], a[pivot] = (a[pivot], a[col])
        v = a[col][col]
        a[col] = [x / v for x in a[col]]
        for r in range(n):
            if r == col:
                continue
            fac = a[r][col]
            if fac:
                a[r] = [x - fac * y for x, y in zip(a[r], a[col])]
    return [r[-1] for r in a]

def matmul(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(len(B)))) for j in range(len(B[0]))] for i in range(len(A))]

def exact_model(config_path: Path):
    c = json.loads(config_path.read_text(), parse_float=F)
    P = [[[F(x) for x in row] for row in c['model'][p]] for p in ['P0', 'P1']]
    actions = list(itertools.product(c['model']['inputs'], c['model']['durations']))
    target = c['model']['target_indicator']
    Q = [[None] * 10 for _ in range(4)]
    O = [[F(0)] * 10 for _ in range(4)]
    G = [[F(0)] * 10 for _ in range(4)]
    for a, (u, t) in enumerate(actions):
        power = [[F(i == j) for j in range(4)] for i in range(4)]
        for _ in range(t):
            power = matmul(power, P[u])
            for i in range(4):
                O[i][a] += sum((power[i][j] * target[j] for j in range(4)))
        for i in range(4):
            Q[i][a] = power[i]
            G[i][a] = O[i][a] - F(c['objectives']['control_cost']) * u * t
    return {'P': P, 'Q': Q, 'O': O, 'G': G, 'actions': actions, 'target': target, 'budget': F(c['objectives']['budget'])}

def evaluate(pi, m):
    if len(pi) != 4 or any((len(row) != 10 for row in pi)):
        raise ValueError('policy must have shape4x10')
    for row in pi:
        total = sum(row)
        if isinstance(total, I):
            if not total.contains(1) or any((I.of(v).lo < 0 for v in row)):
                raise ValueError('invalid interval policy')
        elif total != 1 or min(row) < 0:
            raise ValueError('invalid exact policy')
    n = 4
    PP = [[sum((pi[i][a] * m['Q'][i][a][j] for a in range(10))) for j in range(4)] for i in range(4)]
    A = [[int(i == j) - PP[i][j] for j in range(4)] for i in range(4)]
    cof = [determinant([[A[r][c] for c in range(4) if c != i] for r in range(4) if r != i]) for i in range(4)]
    mass = sum(cof)
    if mass.lo <= 0 if isinstance(mass, I) else mass <= 0:
        raise ArithmeticError('stationary cofactor normalization not positive')
    d = [v / mass for v in cof]
    T = sum((d[i] * pi[i][a] * m['actions'][a][1] for i in range(4) for a in range(10)))
    y = [[d[i] * pi[i][a] / T for a in range(10)] for i in range(4)]
    metrics = {'occupancy': sum((y[i][a] * m['O'][i][a] for i in range(4) for a in range(10))), 'reward': sum((y[i][a] * m['G'][i][a] for i in range(4) for a in range(10))), 'rate': 1 / T, 'mean_duration': T, 'exposure_rate': sum((y[i][a] * u * t for i in range(4) for a, (u, t) in enumerate(m['actions'])))}
    return {'P': PP, 'stationary': d, 'tree_cofactors': cof, 'occupation_measure': y, 'metrics': metrics}

def witness_policy(p: F):
    p = F(p)
    r = (1 - p) / p
    if not F(1, 2) < p < F(3, 4):
        raise ValueError('declared witness range violated')
    low = exp_negative(F(15)) / eighth_root_unit(r)
    den = 1 + I.of(r) + 8 * low
    row = [low / den for _ in range(10)]
    row[2] = 1 / den
    row[3] = I.of(r) / den
    q = exp_negative(F(40, 3))
    den1 = 1 + 9 * q
    pi = [row]
    for a in [5, 6, 6]:
        rrow = [q / den1 for _ in range(10)]
        rrow[a] = 1 / den1
        pi.append(rrow)
    return pi

def stored_policy(W, temp):
    temp = F(temp)
    if temp <= 0:
        raise ValueError('nonpositive temperature')
    out = []
    for row in W:
        row = [F(x) for x in row]
        top = max(row)
        vals = [exp_negative((top - x) / temp) for x in row]
        den = sum(vals)
        out.append([v / den for v in vals])
    return out

def frac_text(x):
    return str(F(x))
