"""Replay past RFQs: compare VendorPick's #1 with the 'lowest quote wins' rule of thumb on what vendors really delivered."""
import pandas as pd

from scoring import Requirement, rank_vendors
from validation import split_rfqs

TRUE_REWORK = 2.0      # what a defective unit really cost (replacement, handling, damaged product)
TRUE_LATE_COST = 0.08  # what a late delivery really cost, as a share of order value


def realised_cost(row, qty):
    value = row["price_per_unit"] * qty
    return (value + row["freight_total"] + row["actual_defect_pct"] / 100 * qty * row["price_per_unit"] * TRUE_REWORK
            + (0 if row["delivered_on_time"] else value * TRUE_LATE_COST))


def evaluate(df: pd.DataFrame, weights: dict):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    need = {"actual_defect_pct", "delivered_on_time"}
    if not need <= set(df.columns):
        raise ValueError("The file needs actual_defect_pct and delivered_on_time columns (what each vendor really delivered).")
    rfqs, problems, fatal = split_rfqs(df)
    if fatal:
        raise ValueError(fatal)
    truth = {(str(r["rfq_id"]), str(r["vendor"]).strip()): r for r in df.to_dict("records")}
    recs = []
    for q in rfqs:
        req = Requirement(q["item"], q["quantity"], q["max_lead_days"], q["required_cert"])
        res, info = rank_vendors(q["vendors"], req, weights)
        elig = [r for r in res if r.eligible]
        if len(elig) < 2:
            continue
        costs = {r.vendor: realised_cost(truth[(q["rfq_id"], r.vendor)], q["quantity"]) for r in elig}
        best = min(costs, key=costs.get)
        pick, cheap = elig[0].vendor, info["cheapest_eligible"]
        recs.append({"rfq": q["rfq_id"], "item": q["item"], "vendorpick": pick, "lowest_quote": cheap, "best_in_hindsight": best,
                     "vp_cost": costs[pick], "lq_cost": costs[cheap], "best_cost": costs[best],
                     "vp_late": 1 - truth[(q["rfq_id"], pick)]["delivered_on_time"],
                     "lq_late": 1 - truth[(q["rfq_id"], cheap)]["delivered_on_time"],
                     "vp_new_vendor": int(truth[(q["rfq_id"], pick)]["years_with_us"] == 0)})
    d = pd.DataFrame(recs)
    if d.empty:
        raise ValueError("No RFQ had two or more eligible vendors to compare.")
    m = {"rfqs": len(d),
         "vp_overrun": (d.vp_cost.sum() / d.best_cost.sum() - 1), "lq_overrun": (d.lq_cost.sum() / d.best_cost.sum() - 1),
         "vp_best": int((d.vendorpick == d.best_in_hindsight).sum()), "lq_best": int((d.lowest_quote == d.best_in_hindsight).sum()),
         "vp_late": int(d.vp_late.sum()), "lq_late": int(d.lq_late.sum()),
         "same_pick": int((d.vendorpick == d.lowest_quote).sum()),
         "saving": float(d.lq_cost.sum() - d.vp_cost.sum()),
         "vp_new_vendor_picks": int(d.vp_new_vendor.sum()),
         "skipped": len(problems)}
    return d, m
