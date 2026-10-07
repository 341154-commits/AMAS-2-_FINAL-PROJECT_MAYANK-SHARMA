"""VendorPick: vendor selection and procurement recommender (use case #6, App format).

Run locally:  python -m streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import ai
from evaluate import evaluate
from scoring import CRITERIA, DEFAULT_WEIGHTS, Requirement, rank_vendors, stability
from validation import VENDOR_COLUMNS, check_requirement, clean_vendors, split_rfqs

BASE = Path(__file__).parent
SAMPLE = BASE / "data" / "sample_rfqs.csv"
HISTORY = BASE / "data" / "history.csv"
SAMPLE_LABELS = {
    "R1": "Shipper boxes: cheap-but-poor vendor, close call, 3 vendors filtered out",
    "R2": "Snack pouches: a brand-new vendor ranks #1",
    "R3": "Trucking: cheapest carrier is too slow",
}
NICE = {"vendor": "Vendor", "price_per_unit": "Price / unit (Rs)", "freight_total": "Freight total (Rs)",
        "defect_rate_pct": "Defect rate %", "on_time_pct": "On-time %", "lead_time_days": "Lead time (days)",
        "payment_terms_days": "Payment terms (days)", "moq": "Min order qty", "monthly_capacity": "Capacity / month",
        "certifications": "Certifications (; separated)", "years_with_us": "Years with us", "notes": "Notes"}

st.set_page_config(page_title="VendorPick · Vendor selection", page_icon="🏭", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.vp-title {font-size: 2.1rem; font-weight: 750; letter-spacing: -0.02em; margin-bottom: 0;}
.vp-sub {color: #475467; font-size: 1.02rem; margin-top: .2rem; max-width: 800px;}
.vp-note {background:#F2F4F7; border-left: 3px solid #0E7C66; padding:.6rem .9rem; font-size:.88rem; color:#344054; border-radius: 4px;}
.vp-win {font-size: 1.5rem; font-weight: 750;}
</style>
""", unsafe_allow_html=True)


def rs(x) -> str:
    return f"Rs {x:,.0f}"


@st.cache_data(show_spinner=False)
def load_samples():
    rfqs, _, _ = split_rfqs(pd.read_csv(SAMPLE))
    return {r["rfq_id"]: r for r in rfqs}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_memo(facts_json: str):
    return ai.recommendation_memo(json.loads(facts_json))


def vendor_frame(vendors):
    return pd.DataFrame(vendors, columns=VENDOR_COLUMNS)


def build_facts(req, vendors, ranked, info):
    by = {v["vendor"]: v for v in vendors}
    out = []
    for r in ranked:
        v = by[r.vendor]
        note = v.get("notes", "")
        note = "[note removed: it contained instructions]" if ai.looks_like_injection(note) else ai.mask(note)
        out.append({"vendor": r.vendor, "rank": r.rank or None, "eligible": r.eligible, "excluded_because": r.reasons_out,
                    "score": r.score, "price_per_unit": v["price_per_unit"], "freight_total": v["freight_total"],
                    "landed_cost": round(r.landed_cost), "expected_tco": round(r.expected_tco),
                    "defect_rate_pct": v["defect_rate_pct"], "on_time_pct": v["on_time_pct"],
                    "lead_time_days": v["lead_time_days"], "payment_terms_days": v["payment_terms_days"],
                    "years_with_us": v["years_with_us"], "strengths": r.strengths, "watch_outs": r.watch_outs,
                    "flags": r.flags, "note": note})
    return {"item": req.item, "quantity": req.quantity, "max_lead_days": req.max_lead_days,
            "required_cert": req.required_cert or "none", "vendors": out, "tco_note": info.get("tco_note", "")}


def rule_memo(req, ranked, info):
    elig = [r for r in ranked if r.eligible]
    if not elig:
        return "No vendor meets the must-have requirements. Relax a requirement or invite more quotes."
    w = elig[0]
    s = (f"Award the {req.quantity:,} x {req.item} order to {w.vendor} (score {w.score:.0f}/100, landed cost {rs(w.landed_cost)}). "
         f"Strengths: {', '.join(w.strengths) or 'balanced across criteria'}.")
    if w.watch_outs:
        s += f" Watch: {', '.join(w.watch_outs).lower()}."
    if len(elig) > 1:
        s += f" Runner-up: {elig[1].vendor} ({elig[1].score:.0f}/100, {rs(elig[1].landed_cost)})."
    excluded = [r for r in ranked if not r.eligible]
    if excluded:
        s += f" {len(excluded)} vendor(s) did not meet the must-haves."
    return s


# ---------------------------------------------------------------- state
ss = st.session_state
samples = load_samples()
first = samples["R1"]
ss.setdefault("editor_ver", 0)
ss.setdefault("vendors_df", vendor_frame(first["vendors"]))
ss.setdefault("req", {"item": first["item"], "quantity": first["quantity"], "max_lead_days": first["max_lead_days"],
                      "required_cert": first["required_cert"]})
ss.setdefault("run", None)
ss.setdefault("quote", None)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Try a sample purchase")
    pick = st.selectbox("Sample", list(samples), label_visibility="collapsed",
                        format_func=lambda k: f"{k} · {samples[k]['rfq_name']} — {SAMPLE_LABELS.get(k, '')}")
    if st.button("Load sample", type="primary"):
        r = samples[pick]
        ss.vendors_df = vendor_frame(r["vendors"])
        ss.req = {"item": r["item"], "quantity": r["quantity"], "max_lead_days": r["max_lead_days"],
                  "required_cert": r["required_cert"]}
        ss.editor_ver += 1
        ss.run, ss.quote = None, None
        st.rerun()
    st.divider()
    st.subheader("What matters most?")
    st.caption("Weights are relative; they are rescaled to 100%.")
    weights = {k: st.slider(label, 0, 100, DEFAULT_WEIGHTS[k], 5, key=f"w_{k}") for k, (label, _, _) in CRITERIA.items()}
    if sum(weights.values()) == 0:
        st.error("At least one weight must be above zero. Using the defaults.")
        weights = dict(DEFAULT_WEIGHTS)
    st.divider()
    if ai.get_key():
        st.success("Gemini key found. AI features are on.")
    else:
        st.warning("No Gemini key. Filtering, ranking and export still work; AI features are off.")
    st.caption("VendorPick is an AI-assisted tool. The ranking is a recommendation for a buyer to review, not a purchase decision.")

# ---------------------------------------------------------------- header
st.markdown('<p class="vp-title">🏭 VendorPick</p>', unsafe_allow_html=True)
st.markdown('<p class="vp-sub">Vendor selection for a procurement team. Must-haves filter vendors out, your weights rank the rest, '
            'and Gemini reads quotations and drafts the award memo.</p>', unsafe_allow_html=True)
st.markdown('<p class="vp-note">Demo set-up: <b>Annapurna Snacks Pvt Ltd</b>, a fictional snack maker buying packaging and '
            'transport. All vendors are sample data. When you use AI features, quotation text and vendor figures are sent to '
            'Google Gemini with emails, phone numbers, PAN, GSTIN and account numbers masked.</p>', unsafe_allow_html=True)
st.write("")
tab_cmp, tab_check, tab_how = st.tabs(["Compare vendors", "Does the ranking work?", "How it works & limits"])

# ================================================================ TAB 1
with tab_cmp:
    q = ss.req
    st.markdown("**What are you buying?**")
    c1, c2, c3, c4 = st.columns([3, 1.3, 1.3, 1.5])
    item = c1.text_input("Item *", q["item"])
    quantity = c2.number_input("Quantity", 1, 1_000_000_000, int(q["quantity"]), step=1000)
    max_lead = c3.number_input("Needed within (days)", 1, 365, int(q["max_lead_days"]))
    cert = c4.text_input("Required certification", q["required_cert"], help="e.g. ISO 9001 or FSSAI. Leave blank if none.")
    ss.req = {"item": item, "quantity": quantity, "max_lead_days": max_lead, "required_cert": cert}

    st.markdown("**Vendor quotes** (edit cells, add rows at the bottom, paste a quotation below, or load a CSV)")
    edited = st.data_editor(
        ss.vendors_df, key=f"editor_{ss.editor_ver}", num_rows="dynamic", hide_index=True,
        column_config={k: (st.column_config.TextColumn(NICE[k], width="medium") if k in ("vendor", "certifications", "notes")
                           else st.column_config.NumberColumn(NICE[k], min_value=0)) for k in VENDOR_COLUMNS})

    with st.expander("📋 Add a vendor from a pasted quotation (AI)"):
        txt = st.text_area("Paste the quotation email or text", height=140,
                           placeholder="Dear Sir, we are pleased to quote Rs 17.60 per box, ex-works Pune. Freight extra at Rs 38,000 ...")
        if st.button("Read quotation with AI"):
            try:
                with st.spinner("Reading the quotation…"):
                    ss.quote = ai.read_quotation(txt)
            except ai.AIError as e:
                ss.quote = None
                st.info(str(e))
        qt = ss.quote
        if qt:
            st.caption("Each number below was found word for word in the text. Check them, and fill in what the quotation "
                       "doesn't cover (quality and delivery figures come from your own records).")
            if qt["injection"]:
                st.warning("The quotation contains instructions aimed at the AI (for example 'rank us first'). They were ignored.")
            if qt["dropped"]:
                st.warning("Dropped because the AI's source text was not found in the quotation: " + ", ".join(qt["dropped"]))
            f = qt["fields"]
            val = lambda k, d=0.0: float(f[k]["value"]) if f.get(k) else d
            a1, a2, a3, a4 = st.columns(4)
            n_vendor = a1.text_input("Vendor", qt["vendor"], key="q_v")
            n_price = a2.number_input("Price / unit", 0.0, value=val("price_per_unit"), key="q_p")
            n_freight = a3.number_input("Freight total", 0.0, value=val("freight_total"), key="q_f")
            n_lead = a4.number_input("Lead time (days)", 0, 365, int(val("lead_time_days")), key="q_l")
            b1, b2, b3, b4 = st.columns(4)
            n_terms = b1.number_input("Payment terms (days)", 0, 180, int(val("payment_terms_days")), key="q_t")
            n_moq = b2.number_input("Min order qty", 0, value=int(val("moq")), key="q_m")
            n_cap = b3.number_input("Capacity / month", 0, value=int(val("monthly_capacity")), key="q_c")
            n_cert = b4.text_input("Certifications", "; ".join(qt["certifications"]), key="q_ce")
            c1_, c2_, c3_ = st.columns(3)
            n_def = c1_.number_input("Defect rate % (your records)", 0.0, 100.0, 2.0, key="q_d")
            n_ot = c2_.number_input("On-time % (your records)", 0.0, 100.0, 90.0, key="q_o")
            n_yrs = c3_.number_input("Years with us", 0, 100, 0, key="q_y")
            with st.popover("Where each number came from"):
                for k, v in f.items():
                    st.write(f"**{NICE[k]}**: " + (f'"{v["quote"]}" → {v["value"]:,}' if v else "not stated"))
            if st.button("Add this vendor"):
                new = {"vendor": n_vendor or "New vendor", "price_per_unit": n_price, "freight_total": n_freight,
                       "defect_rate_pct": n_def, "on_time_pct": n_ot, "lead_time_days": n_lead,
                       "payment_terms_days": n_terms, "moq": n_moq, "monthly_capacity": n_cap,
                       "certifications": n_cert, "years_with_us": n_yrs, "notes": "Added from a pasted quotation."}
                ss.vendors_df = pd.concat([edited, pd.DataFrame([new])], ignore_index=True)
                ss.editor_ver += 1
                ss.quote = None
                st.rerun()

    with st.expander("📄 Load vendors from a CSV"):
        st.download_button("Download the CSV template", SAMPLE.read_bytes(), "vendorpick_template.csv", "text/csv")
        up = st.file_uploader("CSV with one row per vendor quote (first RFQ in the file is loaded)", type=["csv"])
        if up is not None and st.button("Load this file"):
            try:
                rfqs, probs, fatal = split_rfqs(pd.read_csv(up))
            except Exception:
                rfqs, probs, fatal = [], [], "This file could not be read as a CSV. Save it from Excel as 'CSV UTF-8'."
            if fatal or not rfqs:
                st.error(fatal or "No valid rows found.")
            else:
                r = rfqs[0]
                ss.vendors_df = vendor_frame(r["vendors"])
                ss.req = {"item": r["item"], "quantity": r["quantity"], "max_lead_days": r["max_lead_days"],
                          "required_cert": r["required_cert"]}
                ss.editor_ver += 1
                ss.run = None
                if probs:
                    st.warning(f"{len(probs)} row(s) skipped: " + " | ".join(probs[:5]))
                st.rerun()

    if st.button("Rank vendors", type="primary"):
        errs = check_requirement(item, quantity, max_lead)
        vendors, problems = clean_vendors(edited.to_dict("records"))
        if errs or not vendors:
            st.error("Fix these before ranking:\n\n" + "\n".join(f"- {e}" for e in errs + ([] if vendors else ["Add at least one vendor."])))
            ss.run = None
        else:
            ss.run = {"vendors": vendors, "problems": problems,
                      "req": {"item": item, "quantity": int(quantity), "max_lead_days": int(max_lead), "required_cert": cert},
                      "memo": None, "memo_error": "", "answer": ""}

    run = ss.run
    if run:
        req = Requirement(**run["req"])
        ranked, info = rank_vendors(run["vendors"], req, weights)
        elig = [r for r in ranked if r.eligible]
        if run["problems"]:
            st.warning(f"{len(run['problems'])} row(s) skipped:\n\n" + "\n".join(f"- {p}" for p in run["problems"]))
        st.divider()
        if not elig:
            st.error("No vendor meets the must-haves (capacity, minimum order, lead time, certification). "
                     "Relax a requirement or invite more quotes.")
        else:
            w = elig[0]
            l, r_ = st.columns([2, 3])
            with l:
                st.caption("Recommended award")
                st.markdown(f'<div class="vp-win">🏆 {w.vendor}</div>', unsafe_allow_html=True)
                st.markdown(f"Score **{w.score:.0f}/100** · landed cost **{rs(w.landed_cost)}** · "
                            f"expected total cost **{rs(w.expected_tco)}**")
                stab = stability(run["vendors"], req, weights)
                if stab:
                    others = ", ".join(f"{k} ({v})" for k, v in stab["challengers"].items())
                    st.caption(f"Stays #1 in {stab['kept']} of {stab['trials']} weight changes (each weight ±10 points)."
                               + (f" Otherwise: {others}." if others else ""))
            with r_:
                for fl in w.flags:
                    st.warning(fl)
                if info.get("tco_note"):
                    st.info(info["tco_note"])
                if info["cheapest_vendor"] != w.vendor:
                    st.caption(f"Lowest quote overall: {info['cheapest_vendor']}"
                               + ("" if info["cheapest_vendor"] == info["cheapest_eligible"] else
                                  f" (filtered out); lowest eligible quote: {info['cheapest_eligible']}") + ".")

            st.markdown("**Shortlist**")
            st.dataframe(pd.DataFrame([{
                "Rank": r.rank, "Vendor": r.vendor, "Score": r.score, "Landed cost": r.landed_cost,
                "Expected total cost": r.expected_tco, "Strengths": ", ".join(r.strengths),
                "Watch-outs": ", ".join(r.watch_outs), "Flags": " ".join(r.flags)} for r in elig]),
                hide_index=True, column_config={"Landed cost": st.column_config.NumberColumn(format="%.0f"),
                                                "Expected total cost": st.column_config.NumberColumn(format="%.0f")})
            st.markdown("**Where each score comes from** (points per criterion)")
            parts = pd.DataFrame({r.vendor: r.parts for r in elig}).T.rename(columns={k: v[0] for k, v in CRITERIA.items()})
            st.bar_chart(parts, horizontal=True)
        excluded = [r for r in ranked if not r.eligible]
        if excluded:
            st.markdown("**Filtered out (did not meet a must-have)**")
            st.dataframe(pd.DataFrame([{"Vendor": r.vendor, "Landed cost": r.landed_cost, "Why": " ".join(r.reasons_out)}
                                       for r in excluded]), hide_index=True,
                         column_config={"Landed cost": st.column_config.NumberColumn(format="%.0f")})

        facts = build_facts(req, run["vendors"], ranked, info)
        if elig:
            st.subheader("Recommendation memo")
            if ai.get_key() and run["memo"] is None and not run["memo_error"]:
                if st.button("Write the memo with AI"):
                    try:
                        with st.spinner("Writing the memo…"):
                            run["memo"] = cached_memo(json.dumps(facts, sort_keys=True, default=str))
                        if run["memo"]["unverified"] or run["memo"]["wrong_vendor"]:
                            cached_memo.clear()  # never serve a discarded memo from the cache
                    except ai.AIError as e:
                        run["memo_error"] = str(e)
                    st.rerun()
            memo = run["memo"]
            if memo and not memo["unverified"] and not memo["wrong_vendor"]:
                st.markdown(memo["memo"])
                if memo["points"]:
                    st.markdown("**Negotiation points**\n" + "\n".join(f"- {p}" for p in memo["points"]))
                st.caption(f"Written by {memo.get('model') or 'Gemini'}. It backs the #1 vendor and every rupee figure was checked.")
            else:
                if memo and memo["wrong_vendor"]:
                    st.warning("The AI memo recommended a different vendor from the ranking, so it was discarded.")
                elif memo and memo["unverified"]:
                    st.warning("The AI memo used figures not in the quotes (" + ", ".join(memo["unverified"]) + "), so it was discarded.")
                if run["memo_error"]:
                    st.info(run["memo_error"])
                st.markdown(f"**Summary (rule-based):** {rule_memo(req, ranked, info)}")

            st.markdown("**Ask about this comparison**")
            qtext = st.text_input("Question", placeholder="e.g. What should I negotiate with the winner?", label_visibility="collapsed")
            if st.button("Ask"):
                try:
                    with st.spinner("Thinking…"):
                        run["answer"] = ai.ask(qtext, facts)
                except ai.AIError as e:
                    run["answer"] = f"⚠️ {e}"
            if run["answer"]:
                st.info(run["answer"])

        export = pd.DataFrame([{"rank": r.rank or "", "vendor": r.vendor, "eligible": r.eligible, "score": r.score,
                                "landed_cost": round(r.landed_cost), "expected_total_cost": round(r.expected_tco),
                                "strengths": "; ".join(r.strengths), "watch_outs": "; ".join(r.watch_outs),
                                "flags": " ".join(r.flags), "excluded_because": " ".join(r.reasons_out)} for r in ranked])
        memo_txt = (run["memo"]["memo"] if run["memo"] and not run["memo"]["unverified"] and not run["memo"]["wrong_vendor"]
                    else rule_memo(req, ranked, info))
        d1, d2 = st.columns(2)
        d1.download_button("Download comparison (CSV)", export.to_csv(index=False).encode(), "vendor_comparison.csv", "text/csv")
        d2.download_button("Download award memo (TXT)", (f"Purchase: {req.quantity:,} x {req.item}\n"
                           f"Weights: {json.dumps(weights)}\n\n{memo_txt}\n").encode(), "award_memo.txt", "text/plain")
        st.caption("Results stay while this tab is open. A browser refresh starts a fresh session, so download what you need.")

# ================================================================ TAB 2
with tab_check:
    st.markdown("A ranking is only useful if the vendors it picks actually deliver. This tab replays **past** purchases where "
                "we know what every vendor really delivered, and compares VendorPick's #1 with the common rule of thumb "
                "**'lowest quote wins'**. The built-in file is 60 **synthetic** past purchases. It uses your current sidebar weights.")
    hist_up = st.file_uploader("Past purchases with outcomes (optional)", type=["csv"], key="hist")
    try:
        hist = pd.read_csv(hist_up) if hist_up is not None else pd.read_csv(HISTORY)
        d, m = evaluate(hist, weights)
    except Exception as e:
        st.error(str(e) if isinstance(e, ValueError) else "This file could not be read as a CSV.")
        d = None
    if d is not None:
        c = st.columns(3)
        c[0].metric("Purchases compared", m["rfqs"])
        c[1].metric("Same vendor picked by both", m["same_pick"])
        c[2].metric("Saved vs lowest quote", rs(m["saving"]))
        st.dataframe(pd.DataFrame({
            "Cost above best-in-hindsight": [f"{m['vp_overrun']:.1%}", f"{m['lq_overrun']:.1%}"],
            "Picked the best vendor in hindsight": [f"{m['vp_best']} of {m['rfqs']}", f"{m['lq_best']} of {m['rfqs']}"],
            "Late deliveries": [m["vp_late"], m["lq_late"]],
        }, index=["VendorPick #1", "Lowest quote wins"]))
        st.caption(f"VendorPick picked a brand-new vendor {m['vp_new_vendor_picks']} time(s). New vendors' quoted quality is "
                   "a claim, not a track record, and in this data they under-delivered. Real cost = price + freight + defects + late deliveries.")
        st.dataframe(d[["rfq", "item", "vendorpick", "lowest_quote", "best_in_hindsight"]], hide_index=True)

# ================================================================ TAB 3
with tab_how:
    st.markdown("""
#### How vendors are ranked
1. **Must-haves filter vendors out:** capacity at least the quantity, minimum order no more than the quantity,
   lead time within the limit, and the required certification. Filtered vendors are listed with the reason.
2. **Each criterion is scaled 0–1 across the eligible vendors** (best = 1, worst = 0): landed cost (price × quantity + freight),
   defect rate, on-time delivery %, lead time and payment terms.
3. **Score = weighted sum × 100**, using your sidebar weights. Ties go to the lower landed cost.
4. **Expected total cost** adds the cost of defects (1.5 × price per defective unit) and late deliveries (5% of order value
   × late share). If the #1 vendor is not the lowest on expected total cost, the app tells you.
5. **Close call** if #1 and #2 are within 3 points; **stability** shows how often #1 survives ±10-point weight changes.

#### Where the AI comes in
- **Reads pasted quotations:** every number must be quoted word for word from the text, or it is dropped. You confirm before adding.
- **Writes the award memo:** it must back the #1 vendor, and every rupee figure is checked against the quotes, or the memo is discarded.
- **Answers questions** about this comparison only. It cannot change the ranking.

#### What VendorPick will not do
- Decide who gets the order: a buyer does.
- Verify a vendor's claims. Quality and delivery figures for **new vendors** are their own claims; the app flags them.
- Follow instructions written in vendor notes or quotations.

#### Privacy and failure
Quotation text and vendor figures go to Google Gemini only when you use AI features, with contact numbers and tax IDs masked.
On Google's free tier, prompts may be used to improve Google's products, so do not paste confidential quotes into this demo.
If Gemini is down, filtering, ranking, stability, validation and downloads all keep working.
""")
