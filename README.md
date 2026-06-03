# Momentum × Own-Volatility — Research Dashboard

Interactive Streamlit dashboard replicating and stress-testing the thesis from
**"Timing the Momentum Factor Using Its Own Volatility."**

Pulls daily Fama-French data live from the Kenneth R. French Data Library and
runs four research modules end-to-end:

1. **Binary threshold filter** — parameter sweep over volatility cutoffs (5%–30%).
2. **Dynamic volatility scaling** — `w_t = σ* / σ_{t-1}` with leverage cap.
3. **Lookback robustness** — 1M / 6M / 12M side-by-side, with turnover proxy.
4. **Fama-French 5-Factor + MomVol** OLS regression with HAC standard errors.

All trailing-vol signals are lagged by one day to prevent lookahead bias.

---

## Live demo

After deploying to Streamlit Community Cloud, paste your app URL here:

> 🔗 **https://<your-app>.streamlit.app**

---

## Local development

```bash
git clone https://github.com/<your-username>/momentum-vol-dashboard.git
cd momentum-vol-dashboard

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at <http://localhost:8501>. The first launch downloads the
Fama-French daily factor archives (a few hundred KB each); subsequent runs hit
the in-process cache for 12 hours.

---

## Deploy to Streamlit Community Cloud

1. **Push to GitHub** (see *Pushing to GitHub* below).
2. Go to <https://share.streamlit.io> and sign in with your GitHub account.
3. Click **New app** and pick:
   - **Repository:** `<your-username>/momentum-vol-dashboard`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **Python version:** 3.11 (matches `runtime.txt`)
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`, boots the
   app, and gives you a permanent `*.streamlit.app` URL.

No secrets or environment variables are required — the app fetches public
Fama-French data over HTTPS.

### Updating the deployed app

Any push to `main` triggers an automatic redeploy. To force a clean rebuild
(e.g. after editing `requirements.txt`), use the **⋯ → Reboot app** menu in
the Streamlit Cloud dashboard.

---

## Pushing to GitHub

From this directory:

```bash
git init
git add .
git commit -m "Initial commit: momentum × own-vol Streamlit dashboard"
git branch -M main

# Create a NEW empty repo on GitHub first (no README, no .gitignore),
# then link it:
git remote add origin https://github.com/<your-username>/momentum-vol-dashboard.git
git push -u origin main
```

If you have the GitHub CLI installed, you can do it in one shot:

```bash
gh repo create momentum-vol-dashboard --public --source=. --remote=origin --push
```

---

## Project layout

```
momentum-vol-dashboard/
├── streamlit_app.py        # main app — Streamlit Cloud auto-detects this
├── requirements.txt        # pinned dependencies
├── runtime.txt             # Python version pin
├── .streamlit/
│   └── config.toml         # theme + server settings
├── .gitignore
├── LICENSE
└── README.md
```

---

## Notes on methodology

- **Returns** are converted from percent to decimal (`0.01 = 1%`).
- **Trailing volatility** is rolling `std(ddof=1)` × √252.
- **Sharpe** uses the Fama-French `RF` series as the daily risk-free leg.
- **Regression** uses Newey-West (HAC) standard errors with a 5-lag Bartlett
  kernel — daily factor residuals are autocorrelated, so vanilla OLS standard
  errors understate uncertainty.
- **Lookahead control** — every signal feeding day *T* uses information through
  day *T-1* (`.shift(1)`).

## License

MIT — see [LICENSE](LICENSE).
