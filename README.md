# Random Forest Model Evaluation: Airline Customer Satisfaction

Predicting passenger satisfaction from survey data, with a Random Forest tuned through a proper train/validation/test workflow and benchmarked against a single Decision Tree.

## Dataset

[Invistico Airline customer satisfaction survey](Invistico_Airline.csv), 129,880 passenger responses, 22 columns:

- **Target:** `satisfaction` (satisfied / dissatisfied), roughly 55/45 split
- **Demographics and trip info:** Age, Customer Type, Type of Travel, Class, Flight Distance, Departure/Arrival Delay
- **14 service ratings**, each on a 0-5 scale: seat comfort, food and drink, inflight wifi, inflight entertainment, online booking and support, baggage handling, checkin service, cleanliness, and so on

The only data quality issue worth noting is 393 missing values (0.3% of rows) in `Arrival Delay in Minutes`. These were filled with the column median rather than the mean, since delay times are heavily right-skewed by a handful of very long outliers, and the rows with missing data didn't show any pattern that would suggest something other than a recording gap.

## Why a three-way split instead of the usual train/test

A normal train/test split is fine if you're training one model with fixed settings. The moment you start trying different hyperparameter combinations and picking the best one based on test performance, the test set stops being a clean estimate of real-world performance. You're effectively letting test data leak into your model selection process, even if you never explicitly trained on it.

The fix is a third split: hold out a validation set, use it to pick hyperparameters, and only touch the test set once, at the very end, with the model already locked in.

- **Train (60%, ~77,900 rows):** used to fit every candidate model during tuning
- **Validation (20%, ~26,000 rows):** used to score each hyperparameter combination and pick the winner
- **Test (20%, ~26,000 rows):** never touched until the final model is chosen, used exactly once for the final report

All three splits are stratified on the target, so the satisfaction rate (about 54.7%) stays consistent across train, validation, and test.

## Hyperparameter tuning approach

`GridSearchCV` normally manages its own internal cross-validation splits, but that gives no control over which exact rows get used for validation. Since we already built a dedicated validation set by hand, we used `PredefinedSplit` to force `GridSearchCV` to train only on the training rows and score only on the validation rows, for every parameter combination in the grid. The test set never enters this process at all.

**Grid searched:**

| Parameter | Values tried |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 10, 20, None |
| `min_samples_leaf` | 1, 5 |

12 combinations total, scored on validation F1.

**Best configuration found:**

| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | None |
| `min_samples_leaf` | 1 |

Validation F1 for this configuration: **0.958**

A couple of patterns worth flagging from the full grid search results (in the notebook): `max_depth=10` was the clear loser across every other setting, costing about 2.5 points of F1 compared to deeper trees. The Random Forest needs the extra depth to actually benefit from averaging across many trees. `n_estimators` made almost no difference between 100 and 200 once depth and leaf size were reasonable, more trees mainly buys stability rather than extra accuracy past a certain point.

## Results: Decision Tree vs Random Forest (test set, never used during tuning)

| Metric | Decision Tree | Random Forest | Improvement |
|---|---|---|---|
| Accuracy | 0.9202 | 0.9519 | +0.0317 |
| Precision | 0.9250 | 0.9645 | +0.0396 |
| Recall | 0.9297 | 0.9469 | +0.0173 |
| F1 Score | 0.9273 | 0.9556 | +0.0283 |

| | Decision Tree | Random Forest |
|---|---|---|
| Train Accuracy | 0.9286 | 1.0000 |
| Test Accuracy | 0.9202 | 0.9519 |
| Train-Test Gap | 0.0083 | 0.0481 |

The Random Forest wins on every single metric, by roughly 2 to 4 points depending on which one you look at. That's a consistent, real improvement, not a case of trading recall for precision or vice versa.

**A note on the train accuracy of 1.0:** at first glance, a model that gets 100% of its own training data right looks like a textbook overfitting problem. For a single Decision Tree, it would be. For a Random Forest with `min_samples_leaf=1`, it's expected, individual trees do memorize their bootstrap sample down to single-passenger leaves. What keeps the overall model honest is that each tree memorizes a different random subset of rows and a different random subset of features. Their individual mistakes don't line up with each other, and averaging across a forest of uncorrelated mistakes is what produces a prediction that generalizes, even though each individual tree is overfit on its own. The number that actually tells you whether the model works is the test accuracy, and the Random Forest's test accuracy beats the Decision Tree's by 3 points despite the larger train-test gap.

## What drives satisfaction

Ranked by Random Forest feature importance:

1. **Inflight entertainment** — the single strongest predictor by a wide margin
2. **Seat comfort** — second clearest driver, core physical comfort still outweighs most digital factors
3. **Ease of online booking** and **online support** — the pre-flight digital experience carries real weight, not just what happens in the cabin
4. **Leg room service, on-board service, food and drink** — a cluster of in-flight service factors that matter, but each individually less than entertainment or seat comfort

## Recommendation for leadership

Prioritize inflight entertainment and seat comfort first, since these carry the most predictive weight on satisfaction outcomes. Online booking and support is the next clear opportunity, and it's typically cheaper to improve than physical cabin upgrades, so it may deliver faster returns relative to cost. Leg room, on-board service, and food and drink matter, but should be treated as secondary levers rather than the centerpiece of a satisfaction improvement initiative.

On model choice: the Random Forest is the right call for production use here. It's harder to explain to a non-technical audience than a single Decision Tree (you lose the readable if-this-then-that decision path), but the consistent gains across every metric justify that trade-off for a model whose job is making accurate predictions, not serving as an interpretability tool.

## Files in this submission

- `random_forest_airline.ipynb` — full notebook, all cells executed, includes GridSearchCV output, metrics, confusion matrices, comparison charts, and feature importance plot
- `Invistico_Airline.csv` — source dataset
- `README.md` — this file
- `requirements.txt` — dependencies to reproduce the notebook

## Reproducing this

```bash
pip install -r requirements.txt
jupyter notebook random_forest_airline.ipynb
```

Grid search takes roughly 2 to 3 minutes on a standard machine. Everything else runs in seconds.
