"""Generate data/history.csv: 60 SYNTHETIC past RFQs with what each vendor actually delivered.

Each vendor row has its quoted figures (what VendorPick sees) and the realised outcome
(actual defect rate and whether the delivery was on time), which VendorPick never sees.
The hidden truth is noisy and new vendors tend to over-promise, so the ranking can be wrong.
Run from the project folder:  python tools/make_history.py
"""
import random
from pathlib import Path

import pandas as pd

random.seed(5)
ITEMS = [("Corrugated shipper box", 150000, 22, 16.0, 20, "ISO 9001"),
         ("Laminate pouch", 400000, 30, 2.0, 25, "FSSAI"),
         ("PET jar 500 ml", 120000, 25, 7.5, 18, "FSSAI"),
         ("Pallet wrap film (kg)", 8000, 14, 145.0, 10, ""),
         ("FTL trip Pune to Delhi", 30, 3, 60000.0, 8, "")]
rows = []
for n in range(60):
    item, qty, maxlead, base, spread, cert = random.choice(ITEMS)
    qty = int(qty * random.uniform(0.7, 1.4))
    for v in range(random.randint(3, 6)):
        years = random.choice([0, 0, 1, 2, 3, 5, 8])
        q_def = round(random.uniform(0.5, 5.0), 1)
        q_ot = round(random.uniform(75, 98))
        price = round(base * random.uniform(1 - spread / 200, 1 + spread / 200), 2)
        # cheaper vendors are a bit worse on average
        q_def = round(max(0.2, min(6, q_def + (base - price) / base * 20)), 1)
        # new vendors over-promise: real performance is worse than quoted
        over = 1.8 if years == 0 else 1.0
        real_def = max(0.0, round(q_def * over * random.lognormvariate(0, 0.35), 2))
        real_ontime = int(random.random() < (q_ot / 100) / (1.15 if years == 0 else 1.0))
        rows.append(dict(rfq_id=f"P{n+1:03d}", rfq_name=f"Past order {n+1}", item=item, quantity=qty,
                         max_lead_days=maxlead, required_cert=cert, vendor=f"Vendor {n+1}-{chr(65+v)}",
                         price_per_unit=price, freight_total=round(qty * price * random.uniform(0.005, 0.02)),
                         defect_rate_pct=q_def, on_time_pct=q_ot,
                         lead_time_days=random.randint(max(1, maxlead - 10), maxlead + 4),
                         payment_terms_days=random.choice([15, 30, 45, 60]),
                         moq=int(qty * random.choice([0.1, 0.25, 0.5])),
                         monthly_capacity=int(qty * random.choice([0.8, 1.2, 1.5, 2.5])),
                         certifications=cert if (not cert or random.random() < 0.85) else "",
                         years_with_us=years, notes="",
                         actual_defect_pct=real_def, delivered_on_time=real_ontime))
out = Path(__file__).resolve().parents[1] / "data" / "history.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", len(rows), "vendor quotes across 60 past RFQs")
