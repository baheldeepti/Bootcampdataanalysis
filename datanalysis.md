## Introduction

#In this project, we explore the causal impact of attending a coding bootcamp on annual salaries.
#Rather than relying on naive correlations, we apply causal inference techniques to properly estimate the treatment effect while adjusting for confounding factors.
#We simulate an observational dataset where individuals self-select into bootcamps based on prior experience and education level.

#Concepts used:
#- Potential Outcomes Framework
#- Confounders and Selection Bias
#- Propensity Score Matching (PSM)
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/baheldeepti/Bootcampdataanalysis/main/causal_inference_bootcamp_dataset.csv')

df.head()
#Visualize the features
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='treatment', data=df, palette="pastel")
plt.title('Treatment Distribution')
plt.show()

sns.barplot(x='education_level', y='treatment', data=df, ci=None, palette="pastel")
plt.title('Treatment by Education Level')
plt.show()

sns.scatterplot(x='prior_experience', y='treatment', data=df, alpha=0.4)
plt.title('Treatment vs Prior Experience')
plt.show()

sns.boxplot(x='treatment', y='salary', data=df, palette="pastel")
plt.title('Naive Salary Comparison')
plt.show()
#Naives Estimation
avg_salary_treated = df[df['treatment'] == 1]['salary'].mean()
avg_salary_untreated = df[df['treatment'] == 0]['salary'].mean()
naive_difference = avg_salary_treated - avg_salary_untreated

print(f"Naive Salary Difference: ${naive_difference:.2f}")
#Propnesity score matching
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

X = pd.get_dummies(df[['age', 'prior_experience', 'education_level', 'region']], drop_first=True)
y = df['treatment']

# Propensity score model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
df['propensity_score'] = model.predict_proba(X)[:, 1]

# Nearest neighbor matching
treated = df[df['treatment'] == 1]
untreated = df[df['treatment'] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(untreated[['propensity_score']])
distances, indices = nn.kneighbors(treated[['propensity_score']])
matched_untreated = untreated.iloc[indices.flatten()]

# ATT calculation
att = treated['salary'].reset_index(drop=True).mean() - matched_untreated['salary'].reset_index(drop=True).mean()
print(f"Estimated Causal Effect (ATT): ${att:.2f}")

# Install causalml first if you haven't
# pip install causalml

# Import libraries
import pandas as pd
import numpy as np
from causalml.inference.tree import UpliftRandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the updated dataset
df = pd.read_csv('https://raw.githubusercontent.com/baheldeepti/Bootcampdataanalysis/main/causal_inference_bootcamp_dataset_with_presalary.csv')

# Prepare features (One-hot encode categorical variables)
X = pd.get_dummies(df[['age', 'prior_experience', 'education_level', 'region']], drop_first=True)

# Define treatment and outcome
treatment = df['treatment'].astype(str)   # IMPORTANT: treatment must be string type
outcome = df['salary']

# Split into train/test for robustness
X_train, X_test, treatment_train, treatment_test, outcome_train, outcome_test = train_test_split(
    X, treatment, outcome, test_size=0.2, random_state=42)

# Set up and fit the Uplift Random Forest
uplift_model = UpliftRandomForestClassifier(
    n_estimators=2000,
    min_samples_leaf=10,
    max_depth=10,
    control_name='0',          # 0 = control group
    random_state=42
)

# Fit the model
uplift_model.fit(X=X_train.values, treatment=treatment_train.values, y=outcome_train.values)

# Predict Individual Treatment Effects (Uplift scores)
treatment_effects = uplift_model.predict(X_test.values)

# Combine results into a dataframe
cf_results = X_test.copy()
cf_results['individual_treatment_effect'] = treatment_effects

# Preview the top individuals with highest predicted gains
top_individuals = cf_results.sort_values(by='individual_treatment_effect', ascending=False).head(10)

print("Top 10 Individuals with Highest Predicted Treatment Effects:")
print(top_individuals)
import matplotlib.pyplot as plt

# Plot histogram
plt.figure(figsize=(10,6))
plt.hist(cf_results['individual_treatment_effect'], bins=30, edgecolor='black')
plt.title('Distribution of Predicted Individual Treatment Effects')
plt.xlabel('Predicted Treatment Effect')
plt.ylabel('Number of Individuals')
plt.grid(True)
plt.show()
# Plot Top 10 individuals
top_10 = cf_results.sort_values(by='individual_treatment_effect', ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.barh(top_10.index.astype(str), top_10['individual_treatment_effect'], color='skyblue')
plt.xlabel('Predicted Treatment Effect')
plt.title('Top 10 Individuals with Highest Predicted Gains')
plt.gca().invert_yaxis()  # Highest on top
plt.grid(True)
plt.show()
 #Uplift curve
 import numpy as np
import matplotlib.pyplot as plt

# Sort individuals by predicted treatment effect descending
cf_results_sorted = cf_results.sort_values(by='individual_treatment_effect', ascending=False)

# Assume if we treat individuals with high predicted uplift, we get real outcome gains
# Here for visualization we simulate that "true uplift" = observed salary (as proxy)

# Create cumulative sums
cumulative_actual = np.cumsum(outcome_test.loc[cf_results_sorted.index].values)
cumulative_random = np.linspace(0, cumulative_actual[-1], len(cumulative_actual))

# Plot Uplift Curve
plt.figure(figsize=(10,6))
plt.plot(np.arange(len(cumulative_actual)) / len(cumulative_actual), cumulative_actual, label='Model-based Targeting', color='blue')
plt.plot(np.arange(len(cumulative_random)) / len(cumulative_random), cumulative_random, label='Random Targeting', linestyle='--', color='gray')
plt.title('Uplift Curve')
plt.xlabel('Proportion of Population Targeted')
plt.ylabel('Cumulative Outcome (e.g., Salary)')
plt.legend()
plt.grid(True)
plt.show()
# check auc score
from sklearn.metrics import roc_auc_score

# To calculate Uplift Gini, we need "true uplift labels"
# We can create proxy uplift labels based on actual outcome for now (not perfect but good for illustration)

# Create a pseudo-label: if salary above median => positive uplift (treated better), else negative uplift
median_salary = outcome_test.median()

# True uplift labels: 1 = high salary (benefit), 0 = low salary (no benefit)
true_uplift_labels = (outcome_test > median_salary).astype(int)

# Get model scores: predicted individual treatment effects
predicted_uplift_scores = cf_results.loc[outcome_test.index, 'individual_treatment_effect'].values

# Calculate uplift AUC
uplift_auc = roc_auc_score(true_uplift_labels, predicted_uplift_scores)

# Gini = 2*AUC - 1
uplift_gini = 2 * uplift_auc - 1

print(f"Uplift AUC: {uplift_auc:.4f}")
print(f"Uplift Gini Coefficient: {uplift_gini:.4f}")
