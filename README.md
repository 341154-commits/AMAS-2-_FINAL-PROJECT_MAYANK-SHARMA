# VendorPick: vendor selection and procurement recommender

End-term project, use case #6 (Vendor selection or procurement recommender, App format).

A buyer enters what they are buying and the vendor quotes. Must-haves (capacity, minimum order, lead time,
certification) filter vendors out; a weighted score the buyer controls (landed cost, defect rate, on-time delivery,
lead time, payment terms) ranks the rest, with a reason for every point. Google Gemini reads pasted quotations
(every number must be quoted from the text), writes the award memo (it must back the #1 vendor; every rupee figure
is checked) and answers questions about the comparison.

Demo company: **Annapurna Snacks Pvt Ltd** (fictional). All data is invented.

## Files
| File | What it does |
|---|---|
| `app.py` | The Streamlit app (3 tabs) |
| `scoring.py` | Must-have filters, weighted scoring, expected total cost, close-call and stability checks |
| `ai.py` | Gemini calls: quotation reader, award memo, Q&A, masking, verification, fallbacks |
| `validation.py` | Input checks for the vendor table and CSV uploads |
| `evaluate.py` | Replays past purchases: VendorPick's #1 vs "lowest quote wins" |
| `data/sample_rfqs.csv` | 3 demo purchases (14 vendor quotes) with edge cases; also the CSV template |
| `data/history.csv` | 60 synthetic past purchases (265 quotes) with what each vendor really delivered |
| `tools/make_history.py` | Script that generated the synthetic history |

## Run it on your laptop (Windows)
1. Install Python 3.11 or 3.12 from python.org (tick "Add Python to PATH").
2. Unzip `vendorpick.zip`, open the `vendorpick` folder, click the address bar, type `cmd`, press Enter.
3. `python -m pip install -r requirements.txt`
4. `set GEMINI_API_KEY=paste-your-key-here`   (get one at aistudio.google.com → Get API key)
5. `python -m streamlit run app.py`  → opens at http://localhost:8501. Stop with Ctrl + C.

On Mac: use `python3` and `export GEMINI_API_KEY=...`.

## Put it online (free, browser only)
1. github.com/new → name `vendorpick` → Public → Create repository.
2. Click "uploading an existing file" → drag in everything inside this folder (including the `data` folder) → Commit changes.
3. share.streamlit.io → Create app → Deploy a public app from GitHub →
   GitHub URL: `https://github.com/YOUR-USERNAME/vendorpick/blob/main/app.py`
4. Advanced settings → Secrets → `GEMINI_API_KEY = "your-key"` → Deploy.
Never put the key in any file in this folder.

## Demo video script (about 5 minutes)
| Time | Do this | Say this |
|---|---|---|
| 0:00 | Home screen | "VendorPick, use case 6. Must-haves filter vendors out, my weights rank the rest, and Gemini reads quotes and drafts the award memo." |
| 0:20 | Sample R1 is loaded → Rank vendors | "Six quotes for 2 lakh shipper boxes. Three are filtered out with reasons: lead time too long, capacity too small, no ISO 9001. Shree Packaging wins, but it's a close call with Deccan Boxes." |
| 1:00 | Point at the bar chart and the blue note | "Each score is broken down by criterion. And the app warns that Kaveri, the cheapest, still has the lowest expected total cost once defects are priced in, so I should check my weights." |
| 1:30 | Move the Landed cost slider from 35 to 45 | "With a bit more weight on cost, Deccan takes #1. The stability line said this could happen: Shree stays #1 in 8 of 10 weight changes." |
| 2:00 | Write the memo with AI | "Gemini writes the award memo. It must back the #1 vendor, and every rupee figure is checked against the quotes, or it is thrown away." |
| 2:30 | Ask: "Write me a poem" then "What should I negotiate with Shree?" | "Off-topic questions are refused; questions about this comparison are answered from the facts." |
| 2:55 | Paste a quotation in the expander → Read quotation with AI → Add this vendor | "The AI pulls out price, freight, terms and capacity, but only numbers it can quote word for word. If the quote says 'rank us first', that's ignored." |
| 3:35 | Load R2 → Rank vendors | "A brand-new vendor ranks #1 on its own claimed figures. The app flags it: ask for a trial order." |
| 4:00 | Does the ranking work? tab | "On 60 past purchases, VendorPick's picks cost 1.2% more than the best vendor in hindsight, against 2.3% for 'lowest quote wins', with fewer late deliveries. But the cheapest quote picked the single best vendor slightly more often." |
| 4:40 | How it works tab | "Rules and weights decide; AI only reads and writes, and everything works if Gemini is down." |
