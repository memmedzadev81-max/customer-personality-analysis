# Customer Personality Analysis

Marketing campaign data on **2,240** customers. Goal: stop spending on people who never respond, and focus budget on high-value segments.

Uses **RFM**, **K-Means**, and statistical tests — not only charts.

> Note: project files live under the `customer-personality-analysis/` folder (nested copy). Open that folder for data, notebook, and images.

---

## Business questions

1. Who are the high-value customers (RFM)?
2. Do unsupervised clusters agree with RFM?
3. Which segments respond to campaigns?
4. What should marketing stop doing?

---

## Dataset

| Item | Detail |
|------|--------|
| Source | Kaggle Customer Personality / marketing campaign |
| Rows | 2,240 |
| File | `customer-personality-analysis/data/marketing_campaign.csv` |

Cleaning highlights: median income for missing values, age from birth year, messy marital labels fixed, constant columns dropped. Features built: total spend, children flags, campaign accepts, tenure.

---

## Stack

Python · pandas · scikit-learn (KMeans) · scipy · matplotlib · seaborn · Jupyter

---

## Main findings

| Finding | Number |
|---------|--------|
| Overall campaign response | **~14.9%** |
| Wine share of spend | **~50%** |
| Champions | **~24%** of customers → **~49%** of revenue |
| Response: Champions vs Lost | **~29%** vs **~1%** |
| Income vs spend | r ≈ **0.81** |
| Web visits vs web purchases | r ≈ **-0.06** (visits do not mean sales) |

RFM and K-Means point the same way: a high-income / high-spend group that responds, and low-value groups that do not.

---

## Recommendations

1. Target Champions and Loyal; stop mass-blasting Lost / At-Risk
2. Build offers around wine (half of revenue)
3. Low-cost win-back for At Risk before they go Lost
4. Prefer childless, higher-income households for premium campaigns
5. Fix web conversion — more visits without purchases is wasted traffic

---

## How to run

```bash
git clone https://github.com/memmedzadev81-max/customer-personality-analysis.git
cd customer-personality-analysis/customer-personality-analysis
pip install -r requirements.txt
python analysis.py
```

---

## vs other projects

| Repo | Difference |
|------|------------|
| `Ecommerce-Retail-Sales-Analysis` | RFM skeleton on order data (needs your CSV) |
| **This repo** | Full RFM + K-Means + campaign response on marketing dataset |
| `Telco-Customer-Churn` | Churn, not campaign targeting |

---

Vusal Mammadzade · [memmedzadev81-max](https://github.com/memmedzadev81-max)
