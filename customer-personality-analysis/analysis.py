"""
Customer Personality Analysis (Senior-Level)
Author: Vusal Mammadzade (GitHub: memmedzadev81-max)
Cleaning -> Feature Engineering -> EDA -> RFM + K-Means Segmentation
-> Campaign Response Analysis -> Statistical Tests -> Business Insights
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
IMG = "images/"
PAL = sns.color_palette("Set2")

# ======================================================================
# 1. LOAD
# ======================================================================
df = pd.read_csv("data/marketing_campaign.csv")
print("Raw shape:", df.shape)

# ======================================================================
# 2. DATA CLEANING (documented decisions)
# ======================================================================
# 2.1 Income: 24 missing -> impute with median (robust to the 666,666 outlier)
df["Income"] = df["Income"].fillna(df["Income"].median())
# 2.2 Income outlier: one value of 666,666 is a data error -> cap at 99th pct
cap = df["Income"].quantile(0.99)
n_capped = (df["Income"] > cap).sum()
df.loc[df["Income"] > cap, "Income"] = cap

# 2.3 Age from Year_Birth (data collected ~2014, use 2015 reference)
df["Age"] = 2015 - df["Year_Birth"]
# Remove impossible ages (born 1893/1899 -> >115 yrs)
df = df[df["Age"] < 100]

# 2.4 Marital_Status: merge junk categories
df["Marital_Status"] = df["Marital_Status"].replace(
    {"Alone": "Single", "Absurd": "Single", "YOLO": "Single",
     "Together": "Partner", "Married": "Partner",
     "Divorced": "Single", "Widow": "Single"})

# 2.5 Education: simplify
df["Education"] = df["Education"].replace(
    {"2n Cycle": "Master", "Graduation": "Graduate",
     "Master": "Master", "PhD": "PhD", "Basic": "Undergraduate"})

# 2.6 Date parsing + tenure
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
ref_date = df["Dt_Customer"].max()
df["Tenure_days"] = (ref_date - df["Dt_Customer"]).dt.days

# 2.7 Drop constant / id columns
df = df.drop(columns=["Z_CostContact", "Z_Revenue", "ID", "Year_Birth"])

print("After cleaning:", df.shape)

# ======================================================================
# 3. FEATURE ENGINEERING
# ======================================================================
spend_cols = ["MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df["TotalSpend"] = df[spend_cols].sum(axis=1)
df["Children"] = df["Kidhome"] + df["Teenhome"]
df["HasChildren"] = (df["Children"] > 0).astype(int)
df["TotalPurchases"] = df[["NumWebPurchases", "NumCatalogPurchases",
                           "NumStorePurchases"]].sum(axis=1)
df["TotalAcceptedCmp"] = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                             "AcceptedCmp4", "AcceptedCmp5"]].sum(axis=1)

print("\nKey averages:")
print(f"  Avg age: {df['Age'].mean():.0f}")
print(f"  Avg income: ${df['Income'].mean():,.0f}")
print(f"  Avg total spend (2y): ${df['TotalSpend'].mean():,.0f}")
print(f"  Last-campaign response rate: {df['Response'].mean()*100:.1f}%")

# ======================================================================
# 4. EDA
# ======================================================================
# Fig 1: Spend by product category
fig, ax = plt.subplots(figsize=(8.5, 4.5))
cat_spend = df[spend_cols].sum().sort_values()
labels = [c.replace("Mnt", "").replace("Products", "") for c in cat_spend.index]
bars = ax.barh(labels, cat_spend.values, color=PAL[0])
ax.set_title("Total Spend by Product Category", weight="bold")
ax.set_xlabel("Total spend ($)")
for b, v in zip(bars, cat_spend.values):
    ax.text(v, b.get_y()+b.get_height()/2, f" ${v/1000:.0f}K", va="center", fontsize=9)
plt.savefig(IMG + "01_spend_by_category.png"); plt.close()

# Fig 2: Income vs Spend (key relationship)
fig, ax = plt.subplots(figsize=(7.5, 5))
sc = ax.scatter(df["Income"], df["TotalSpend"], c=df["Age"],
                cmap="viridis", alpha=0.6, s=22)
ax.set_title("Income vs Total Spend (colored by Age)", weight="bold")
ax.set_xlabel("Income ($)"); ax.set_ylabel("Total spend ($)")
plt.colorbar(sc, label="Age")
plt.savefig(IMG + "02_income_vs_spend.png"); plt.close()

# Fig 3: Spend distribution by children
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=df, x="Children", y="TotalSpend", hue="Children",
            palette="Set2", legend=False, ax=ax)
ax.set_title("Total Spend by Number of Children", weight="bold")
plt.savefig(IMG + "03_spend_by_children.png"); plt.close()

# Fig 4: Purchase channels
fig, ax = plt.subplots(figsize=(7, 4.5))
chan = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum()
chan.index = ["Web", "Catalog", "Store"]
bars = ax.bar(chan.index, chan.values, color=PAL[:3])
ax.set_title("Purchases by Channel", weight="bold"); ax.set_ylabel("Total purchases")
for b, v in zip(bars, chan.values):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:,.0f}", ha="center", va="bottom")
plt.savefig(IMG + "04_channels.png"); plt.close()

# ======================================================================
# 5. RFM SEGMENTATION
# ======================================================================
# Recency = df['Recency'] (days since last purchase)
# Frequency = TotalPurchases ; Monetary = TotalSpend
rfm = df[["Recency", "TotalPurchases", "TotalSpend"]].copy()
rfm.columns = ["Recency", "Frequency", "Monetary"]
# Score 1-4 (quartiles). Recency reversed (lower=better)
rfm["R"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["M"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
rfm["RFM_Score"] = rfm[["R", "F", "M"]].sum(axis=1)

def rfm_segment(s):
    if s >= 10: return "Champions"
    if s >= 8:  return "Loyal"
    if s >= 6:  return "Potential"
    if s >= 4:  return "At Risk"
    return "Lost"
rfm["Segment"] = rfm["RFM_Score"].apply(rfm_segment)
df["RFM_Segment"] = rfm["Segment"]

print("\nRFM segment sizes:")
print(rfm["Segment"].value_counts())

# Fig 5: RFM segments
fig, ax = plt.subplots(figsize=(8, 4.5))
order = ["Champions", "Loyal", "Potential", "At Risk", "Lost"]
seg = rfm["Segment"].value_counts().reindex(order)
bars = ax.bar(seg.index, seg.values, color=sns.color_palette("RdYlGn_r", 5)[::-1])
ax.set_title("Customer Segments (RFM)", weight="bold"); ax.set_ylabel("Customers")
for b, v in zip(bars, seg.values):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v}", ha="center", va="bottom")
plt.savefig(IMG + "05_rfm_segments.png"); plt.close()

# ======================================================================
# 6. K-MEANS CLUSTERING (Income + Spend + Age)
# ======================================================================
feat = df[["Income", "TotalSpend", "Age"]].copy()
X = StandardScaler().fit_transform(feat)
# Elbow already known ~4 for this data
km = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = km.fit_predict(X)

cluster_profile = df.groupby("Cluster")[["Income", "TotalSpend", "Age",
                                         "TotalPurchases", "Response"]].mean().round(1)
print("\nK-Means cluster profile:")
print(cluster_profile)

# Fig 6: clusters in Income-Spend space
fig, ax = plt.subplots(figsize=(7.5, 5))
for c in sorted(df["Cluster"].unique()):
    sub = df[df["Cluster"] == c]
    ax.scatter(sub["Income"], sub["TotalSpend"], label=f"Cluster {c}", alpha=0.6, s=22)
ax.set_title("K-Means Clusters: Income vs Spend", weight="bold")
ax.set_xlabel("Income ($)"); ax.set_ylabel("Total spend ($)"); ax.legend()
plt.savefig(IMG + "06_kmeans_clusters.png"); plt.close()

# ======================================================================
# 7. CAMPAIGN RESPONSE ANALYSIS
# ======================================================================
resp_by_seg = df.groupby("RFM_Segment")["Response"].mean().reindex(order) * 100
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(resp_by_seg.index, resp_by_seg.values,
              color=sns.color_palette("RdYlGn_r", 5)[::-1])
ax.set_title("Campaign Response Rate by Segment", weight="bold")
ax.set_ylabel("Response rate (%)")
for b, v in zip(bars, resp_by_seg.values):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f}%", ha="center", va="bottom")
plt.savefig(IMG + "07_response_by_segment.png"); plt.close()

# ======================================================================
# 8. STATISTICAL TESTS
# ======================================================================
print("\n" + "="*60); print("STATISTICAL TESTS"); print("="*60)

# 8.1 T-test: income responders vs non-responders
r = df[df["Response"] == 1]["Income"]; nr = df[df["Response"] == 0]["Income"]
t, p = stats.ttest_ind(r, nr, equal_var=False)
print(f"\nT-test Income (responders vs not): mean {r.mean():.0f} vs {nr.mean():.0f}")
print(f"  t={t:.2f}, p={p:.2e} -> {'SIGNIFICANT' if p<0.05 else 'not sig.'}")

# 8.2 T-test: spend responders vs non
r = df[df["Response"] == 1]["TotalSpend"]; nr = df[df["Response"] == 0]["TotalSpend"]
t, p = stats.ttest_ind(r, nr, equal_var=False)
print(f"\nT-test Spend (responders vs not): mean {r.mean():.0f} vs {nr.mean():.0f}")
print(f"  t={t:.2f}, p={p:.2e} -> {'SIGNIFICANT' if p<0.05 else 'not sig.'}")

# 8.3 ANOVA: spend across education
groups = [g["TotalSpend"].values for _, g in df.groupby("Education")]
f, p = stats.f_oneway(*groups)
print(f"\nANOVA Spend across Education: F={f:.2f}, p={p:.2e} -> {'SIGNIFICANT' if p<0.05 else 'not sig.'}")

# 8.4 Chi-square: response vs having children
table = pd.crosstab(df["HasChildren"], df["Response"])
chi2, p, dof, _ = stats.chi2_contingency(table)
print(f"\nChi-square Response vs HasChildren: chi2={chi2:.2f}, p={p:.2e} -> {'SIGNIFICANT' if p<0.05 else 'not sig.'}")

# 8.5 Correlation: income vs spend
rho, p = stats.pearsonr(df["Income"], df["TotalSpend"])
print(f"\nPearson Income vs Spend: r={rho:.3f}, p={p:.2e}")

# 8.6 Correlation: web visits vs web purchases
rho2, p2 = stats.pearsonr(df["NumWebVisitsMonth"], df["NumWebPurchases"])
print(f"Pearson WebVisits vs WebPurchases: r={rho2:.3f}, p={p2:.2e}")

print("\nDONE.")
