"""Transparent vendor scoring for VendorPick.

Hard filters decide who is eligible. A weighted score ranks the eligible vendors.
The AI never ranks or filters; it only reads quotations and writes the memo.
"""
from dataclasses import dataclass, field
from itertools import product

CRITERIA = {
    # key: (label, higher_is_better, default weight)
    "landed_cost": ("Landed cost", False, 35),
    "defect_rate_pct": ("Quality (defect rate)", False, 25),
    "on_time_pct": ("On-time delivery", True, 20),
    "lead_time_days": ("Lead time", False, 10),
    "payment_terms_days": ("Payment terms", True, 10),
}
DEFAULT_WEIGHTS = {k: v[2] for k, v in CRITERIA.items()}
CLOSE_CALL_POINTS = 3
REWORK_FACTOR = 1.5      # a defective unit costs 1.5x its price (replacement + handling)
LATE_PENALTY = 0.05      # a late delivery costs about 5% of order value (stock-outs, expediting)


@dataclass
class Requirement:
    item: str
    quantity: int
    max_lead_days: int
    required_cert: str = ""          # e.g. "ISO 9001"; empty = none


@dataclass
class VendorResult:
    vendor: str
    eligible: bool
    reasons_out: list = field(default_factory=list)
    landed_cost: float = 0.0
    expected_tco: float = 0.0
    score: float = 0.0
    rank: int = 0
    parts: dict = field(default_factory=dict)      # criterion -> points earned
    strengths: list = field(default_factory=list)
    watch_outs: list = field(default_factory=list)
    flags: list = field(default_factory=list)


def landed_cost(v, req):
    return v["price_per_unit"] * req.quantity + v["freight_total"]


def expected_tco(v, req):
    order_value = v["price_per_unit"] * req.quantity
    defects = req.quantity * v["defect_rate_pct"] / 100 * v["price_per_unit"] * REWORK_FACTOR
    lateness = (1 - v["on_time_pct"] / 100) * order_value * LATE_PENALTY
    return landed_cost(v, req) + defects + lateness


def eligibility(v, req):
    out = []
    if v["monthly_capacity"] < req.quantity:
        out.append(f"Capacity {v['monthly_capacity']:,} is below the {req.quantity:,} needed.")
    if v["moq"] > req.quantity:
        out.append(f"Minimum order {v['moq']:,} is above the {req.quantity:,} needed.")
    if v["lead_time_days"] > req.max_lead_days:
        out.append(f"Lead time {v['lead_time_days']} days is longer than the {req.max_lead_days}-day limit.")
    if req.required_cert:
        certs = [c.strip().lower() for c in str(v.get("certifications", "")).split(";")]
        if req.required_cert.strip().lower() not in certs:
            out.append(f"Does not hold {req.required_cert}.")
    return out


def normalise(values, higher_is_better):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [((x - lo) / (hi - lo)) if higher_is_better else ((hi - x) / (hi - lo)) for x in values]


def rank_vendors(vendors: list, req: Requirement, weights: dict = None):
    """Returns (results sorted by rank, info dict)."""
    weights = weights or DEFAULT_WEIGHTS
    wsum = sum(weights.values()) or 1
    w = {k: weights[k] / wsum for k in CRITERIA}
    results = []
    for v in vendors:
        r = VendorResult(vendor=v["vendor"], eligible=True)
        r.reasons_out = eligibility(v, req)
        r.eligible = not r.reasons_out
        r.landed_cost = landed_cost(v, req)
        r.expected_tco = expected_tco(v, req)
        results.append(r)

    elig = [(r, v) for r, v in zip(results, vendors) if r.eligible]
    if elig:
        cols = {}
        for k, (label, hib, _) in CRITERIA.items():
            vals = [r.landed_cost if k == "landed_cost" else v[k] for r, v in elig]
            cols[k] = normalise(vals, hib)
        for i, (r, v) in enumerate(elig):
            r.parts = {k: round(100 * w[k] * cols[k][i], 1) for k in CRITERIA}
            r.score = round(sum(r.parts.values()), 1)
            for k, (label, _, _) in CRITERIA.items():
                if w[k] == 0:
                    continue
                if cols[k][i] >= 0.8 and len(elig) > 1:
                    r.strengths.append(label)
                elif cols[k][i] <= 0.2 and len(elig) > 1:
                    r.watch_outs.append(label)
        order = sorted(elig, key=lambda x: (-x[0].score, x[0].landed_cost))
        for n, (r, _) in enumerate(order, start=1):
            r.rank = n
    if len(elig) >= 2:
        top = sorted([r for r, _ in elig], key=lambda r: r.rank)
        if top[0].score - top[1].score <= CLOSE_CALL_POINTS:
            for r in top[:2]:
                r.flags.append(f"Close call: #1 and #2 are within {CLOSE_CALL_POINTS} points. A small change in weights or price can swap them.")
    for r, v in zip(results, vendors):
        if v["defect_rate_pct"] >= 3:
            r.flags.append(f"High defect rate ({v['defect_rate_pct']}%).")
        if v["on_time_pct"] < 85:
            r.flags.append(f"Late on {100 - v['on_time_pct']:.0f}% of past deliveries.")
        if v.get("years_with_us", 0) == 0:
            r.flags.append("New vendor: no history with us; ask for a trial order or references.")
    tco_best = min(elig, key=lambda x: x[0].expected_tco)[0] if elig else None
    ranked = sorted([r for r in results if r.eligible], key=lambda r: r.rank) + [r for r in results if not r.eligible]
    cheapest = min(results, key=lambda r: r.landed_cost) if results else None
    info = {"eligible": len(elig), "total": len(results),
            "cheapest_vendor": cheapest.vendor if cheapest else "",
            "cheapest_eligible": min((r for r in results if r.eligible), key=lambda r: r.landed_cost).vendor if elig else "",
            "lowest_tco_eligible": tco_best.vendor if tco_best else ""}
    if elig and ranked[0].vendor != info["lowest_tco_eligible"]:
        info["tco_note"] = (f"{ranked[0].vendor} ranks #1 on your weights, but {info['lowest_tco_eligible']} has the lowest "
                            f"expected total cost once defects and late deliveries are priced in. Check your weights.")
    return ranked, info


def stability(vendors: list, req: Requirement, weights: dict, step: int = 10):
    """Move each weight up and down by `step` points; count how often the #1 vendor stays #1."""
    base, _ = rank_vendors(vendors, req, weights)
    if not base or not base[0].eligible:
        return None
    winner = base[0].vendor
    trials, kept, challengers = 0, 0, {}
    for k, sign in product(CRITERIA, (-1, 1)):
        w2 = dict(weights)
        w2[k] = max(0, w2[k] + sign * step)
        if sum(w2.values()) == 0:
            continue
        res, _ = rank_vendors(vendors, req, w2)
        trials += 1
        if res and res[0].vendor == winner:
            kept += 1
        elif res:
            challengers[res[0].vendor] = challengers.get(res[0].vendor, 0) + 1
    return {"winner": winner, "trials": trials, "kept": kept, "challengers": challengers}
