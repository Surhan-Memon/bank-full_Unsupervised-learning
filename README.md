# Bank Marketing Campaign — KMeans Clustering

An unsupervised machine learning project that applies **KMeans clustering**
to segment bank customers based on their demographic and campaign data,
using the UCI Bank Marketing dataset.

---

## Dataset

- **File:** `bank-full.csv`
- **Size:** 45,211 records, 17 columns
- **Target:** `Target` → `yes` (subscribed) / `no` (did not subscribe to term deposit)

### Key Features

| Feature | Description |
|---------|-------------|
| `age` | Customer age |
| `job` | Job category (12 types) |
| `marital` | Marital status |
| `education` | Education level |
| `default` | Has credit in default? |
| `balance` | Average yearly bank balance |
| `housing` | Has housing loan? |
| `loan` | Has personal loan? |
| `contact` | Contact type (cellular/telephone/unknown) |
| `duration` | Last call duration (seconds) |
| `campaign` | Number of contacts in this campaign |
| `pdays` | Days since last contact from previous campaign |
| `previous` | Number of previous contacts |
| `poutcome` | Previous campaign outcome |

### Notable Stats
Default (no credit issues): 44,396  vs  yes: 815
Personal loan (no):         37,967  vs  yes: 7,244

---

## Exploratory Data Analysis

- **Histplot** of `age` split by `loan` status
- **Histplot** of `pdays` (filtered out -1 sentinel values for "never contacted")
- **Histplot** of `duration` split by `contact` type (limited to 0–1000s range)
- **Count plots** for `contact`, `job` (ordered by frequency), `education` by `default`, `default`
- **Pairplot** across all numeric features

---

## Preprocessing

- **One-hot encoded** all categorical features with `pd.get_dummies()`
  - Expanded from 17 columns → **53 binary/numeric columns**
- **StandardScaler** applied to all features before clustering
  - Clustering is distance-based — scaling is essential

---

## KMeans Clustering

### Initial Model — 2 Clusters

```python
KMeans(n_clusters=2)
cluster_labels = model.fit_predict(scaled_X)
```

- Assigned each customer to **Cluster 0** or **Cluster 1**
- Added `Cluster` column back to the DataFrame
- Plotted **feature correlations with cluster label** (sorted bar chart)
  to identify which features drive the cluster separation

---

## Elbow Method — Finding Optimal K

```python
for k in range(2, 10):
    model = KMeans(n_clusters=k)
    ssd.append(model.inertia_)
```

### Sum of Squared Distances (Inertia) by K

| K | SSD |
|---|-----|
| 2 | 2,282,337 |
| 3 | 2,292,555 |
| 4 | 2,077,977 |
| 5 | 2,032,090 |
| 6 | 2,001,031 |
| 7 | 1,950,634 |
| 8 | 1,840,229 |
| 9 | 1,788,012 |

### Differences (SSD drops between K values)

| K | Drop |
|---|------|
| 3→4 | **-214,578** (largest drop) |
| 4→5 | -45,887 |
| 5→6 | -31,060 |
| 6→7 | -50,396 |
| 7→8 | **-110,405** (second notable drop) |

> The largest single drop occurs at **K=4**, suggesting **4 clusters**
> may be the natural elbow point in this dataset.

- Plotted **SSD vs K** with `'o--'` style for visual elbow identification

---

## Libraries Used

- `numpy`, `pandas` — data handling
- `matplotlib`, `seaborn` — visualization (histplot, countplot, pairplot, bar)
- `scikit-learn`:
  - `KMeans` — clustering
  - `StandardScaler` — feature scaling

---

## Key Takeaways

| Step | Detail |
|------|--------|
| Dataset size | 45,211 customers |
| Features after encoding | 53 columns |
| Initial clusters | 2 |
| Suggested optimal K | **4** (largest elbow drop) |
| Scaling method | StandardScaler (required for KMeans) |

> **Lesson 1:** KMeans is sensitive to scale — always standardize before clustering.
> Features like `balance` (range: -8,019 to 102,127) would dominate
> `age` (18–95) without scaling.

> **Lesson 2:** The elbow method uses the **rate of change** of inertia,
> not just the raw values — `pd.Series(ssd).diff()` makes this easy to spot.

> **Lesson 3:** Clustering an already-labeled dataset lets you compare
> discovered clusters against known outcomes — a powerful validation technique
> even in unsupervised settings.
