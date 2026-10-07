"""Checks that run before anything is ranked or sent to the AI."""
import pandas as pd

VENDOR_COLUMNS = ["vendor", "price_per_unit", "freight_total", "defect_rate_pct", "on_time_pct", "lead_time_days",
                  "payment_terms_days", "moq", "monthly_capacity", "certifications", "years_with_us", "notes"]
RFQ_COLUMNS = ["rfq_id", "rfq_name", "item", "quantity", "max_lead_days", "required_cert"]
RANGES = {
    "price_per_unit": (0.01, 10_000_000), "freight_total": (0, 100_000_000),
    "defect_rate_pct": (0, 100), "on_time_pct": (0, 100), "lead_time_days": (0, 365),
    "payment_terms_days": (0, 180), "moq": (0, 100_000_000), "monthly_capacity": (0, 1_000_000_000),
    "years_with_us": (0, 100),
}
WHOLE = {"lead_time_days", "payment_terms_days", "moq", "monthly_capacity", "years_with_us"}


def _blank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


def check_requirement(item, quantity, max_lead_days):
    errs = []
    if not str(item or "").strip():
        errs.append("Item being bought is required.")
    try:
        if int(quantity) <= 0:
            errs.append("Quantity must be more than zero.")
    except (TypeError, ValueError):
        errs.append("Quantity must be a whole number.")
    try:
        if not 1 <= int(max_lead_days) <= 365:
            errs.append("Maximum lead time must be between 1 and 365 days.")
    except (TypeError, ValueError):
        errs.append("Maximum lead time must be a whole number.")
    return errs


def clean_vendors(rows: list):
    """Returns (clean_rows, problems). Bad rows are skipped and listed, never guessed."""
    clean, problems, seen = [], [], {}
    for i, raw in enumerate(rows, start=1):
        name = "" if _blank(raw.get("vendor")) else str(raw["vendor"]).strip()
        if not name and all(_blank(raw.get(k)) for k in RANGES):
            continue  # empty editor row
        errs, v = [], {"vendor": name}
        if not name:
            errs.append("vendor name is required")
        for k, (lo, hi) in RANGES.items():
            val = raw.get(k)
            if _blank(val):
                if k in ("freight_total", "years_with_us"):
                    v[k] = 0
                    continue
                errs.append(f"{k.replace('_', ' ')} is missing")
                continue
            try:
                x = float(val)
            except (TypeError, ValueError):
                errs.append(f"{k.replace('_', ' ')} must be a number")
                continue
            if not lo <= x <= hi:
                errs.append(f"{k.replace('_', ' ')} must be between {lo:,} and {hi:,} (got {x:,})")
                continue
            v[k] = int(round(x)) if k in WHOLE else x
        v["certifications"] = "" if _blank(raw.get("certifications")) else str(raw["certifications"]).strip()
        v["notes"] = "" if _blank(raw.get("notes")) else str(raw["notes"]).strip()[:500]
        key = name.lower()
        if key and key in seen:
            errs.append(f"duplicate of the vendor on row {seen[key]}")
        if errs:
            problems.append(f"Row {i} ({name or 'no name'}): " + "; ".join(errs))
            continue
        seen[key] = i
        clean.append(v)
    return clean, problems


def split_rfqs(df: pd.DataFrame):
    """Turn a multi-RFQ CSV into RFQs. Returns (rfqs, problems, fatal)."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in RFQ_COLUMNS + VENDOR_COLUMNS if c not in df.columns and c not in ("notes", "years_with_us", "required_cert")]
    if missing:
        return [], [], "The file is missing these columns: " + ", ".join(missing) + ". Download the template."
    if df.empty:
        return [], [], "The file has headers but no vendors."
    rfqs, problems = [], []
    for rid, g in df.groupby("rfq_id", sort=False):
        f = g.iloc[0]
        errs = check_requirement(f["item"], f["quantity"], f["max_lead_days"])
        if errs:
            problems.append(f"RFQ {rid}: " + "; ".join(errs))
            continue
        vendors, vp = clean_vendors(g.to_dict("records"))
        problems += [f"RFQ {rid}, {p}" for p in vp]
        if vendors:
            rfqs.append({"rfq_id": str(rid), "rfq_name": str(f["rfq_name"]), "item": str(f["item"]),
                         "quantity": int(f["quantity"]), "max_lead_days": int(f["max_lead_days"]),
                         "required_cert": "" if _blank(f.get("required_cert")) else str(f["required_cert"]).strip(),
                         "vendors": vendors})
    return rfqs, problems, ""
