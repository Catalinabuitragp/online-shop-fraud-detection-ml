# ==========================================================
# Fraud Detection Using Machine Learning
# Artificial Intelligence - Week 5
# ==========================================================


# %%
# 1. Import libraries

import pandas as pd
import matplotlib.pyplot as plt


# %%
# 2. Load dataset

file_path = (
    "/Users/catalinabuitragotorres/Documents/"
    "Potomac University/Artificial Intelligence/week 5/"
    "Fraudulent_online_shops_dataset.csv"
)

df = pd.read_csv(file_path)


# Clean column names
# Example: "Number  of digits" becomes "Number of digits"

df.columns = (
    df.columns
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)


# %%
# 3. Display basic dataset information

print("\n==============================")
print("Basic Dataset Information")
print("==============================")

print("\nFirst five rows:")
print(df.head())

print("\nDataset dimensions:")
print(df.shape)

print("\nDataset information:")
df.info()


# %%
# 4. Exploratory Data Analysis

print("\n==============================")
print("Exploratory Data Analysis")
print("==============================")

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())


# %%
# 5. Class distribution

print("\n==============================")
print("Class Distribution")
print("==============================")

print("\nNumber of observations by class:")
print(df["Label"].value_counts())

print("\nPercentage by class:")
print(
    df["Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# %%
# 6. Plot class distribution

plt.figure(figsize=(6, 4))

df["Label"].value_counts().plot(
    kind="bar"
)

plt.title("Fraudulent vs. Legitimate Online Shops")
plt.xlabel("Class")
plt.ylabel("Number of Shops")
plt.xticks(rotation=0)
plt.tight_layout()

# Save the graph instead of stopping the script with plt.show()
plot_path = (
    "/Users/catalinabuitragotorres/Documents/"
    "Potomac University/Artificial Intelligence/week 5/"
    "class_distribution.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nClass distribution graph saved at:")
print(plot_path)


# %%
# 7. Analyze categorical variables

print("\n==============================")
print("Categorical Variables Analysis")
print("==============================")

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for column in categorical_columns:

    print("\n-----------------------------------")
    print(column)
    print("-----------------------------------")

    print("Unique values:", df[column].nunique())

    print(
        df[column]
        .value_counts(dropna=False)
        .head(10)
    )


# %%
# 8. Analyze domain registration dates

print("\n==============================")
print("Domain Registration Date Analysis")
print("==============================")

# Keep the original column as text for counting Hidden values
registration_column = (
    df["Domain registration date"]
    .astype("string")
    .str.strip()
)

# Count values explicitly marked as Hidden
hidden_count = (
    registration_column
    .str.lower()
    .eq("hidden")
    .sum()
)

# Count actual missing values
missing_count = (
    df["Domain registration date"]
    .isna()
    .sum()
)

total_unavailable = hidden_count + missing_count

print("\nHidden registration dates:")
print(hidden_count)

print("\nMissing registration dates:")
print(missing_count)

print("\nTotal unavailable registration dates:")
print(total_unavailable)


# Replace Hidden and blank values with missing values

registration_dates_text = registration_column.replace(
    {
        "Hidden": pd.NA,
        "hidden": pd.NA,
        "": pd.NA
    }
)


# Convert valid values to datetime

registration_dates = pd.to_datetime(
    registration_dates_text,
    errors="coerce",
    format="mixed"
)


print("\nValid registration dates:")
print(registration_dates.notna().sum())

print("\nUnavailable or invalid registration dates:")
print(registration_dates.isna().sum())

print("\nEarliest registration date:")
print(registration_dates.min())

print("\nLatest registration date:")
print(registration_dates.max())


# %%
# 9. Define the historical reference date

# The latest observed domain registration occurred in August 2023.
# September 1, 2023 is selected because it is the first full-month
# boundary immediately after the latest observed registration.

reference_date = pd.Timestamp("2023-09-01")

print("\n==============================")
print("Reference Date")
print("==============================")

print("\nSelected reference date:")
print(reference_date)

print("\nLatest observed registration date:")
print(registration_dates.max())

print(
    "\nDays between the latest registration "
    "and the reference date:"
)

print(
    (
        reference_date
        - registration_dates.max()
    ).days
)


# Check that no valid registration date occurs after the reference date

dates_after_reference = (
    registration_dates > reference_date
).sum()

print("\nRegistration dates after the reference date:")
print(dates_after_reference)


# %%
# 10. Data preprocessing and feature engineering

print("\n==============================")
print("Data Preprocessing")
print("==============================")

# Create a copy so the original dataset remains unchanged

df_clean = df.copy()


# Remove the original URL because it is a unique identifier

df_clean = df_clean.drop(
    columns=["Online shop URL"]
)


# Convert the target variable to binary values
# 1 = fraudulent
# 0 = legitimate

df_clean["Label"] = (
    df_clean["Label"]
    .map({
        "legitimate": 0,
        "fraudulent": 1
    })
)


# Replace missing TrustPilot scores with -1
# The dataset already uses -1 to indicate that no score is available

df_clean["TrustPilot score"] = (
    df_clean["TrustPilot score"]
    .fillna(-1)
)


# Create a binary indicator for unavailable registration dates
# 1 = date was hidden, missing, or invalid
# 0 = date was available

df_clean["Domain registration unavailable"] = (
    registration_dates
    .isna()
    .astype(int)
)


# Calculate domain age at the historical reference date

df_clean["Domain age years"] = (
    (
        reference_date
        - registration_dates
    ).dt.days
    / 365.25
)


# Confirm that there are no negative domain ages

negative_domain_ages = (
    df_clean["Domain age years"] < 0
).sum()

print("\nNegative domain ages:")
print(negative_domain_ages)


# Replace unavailable domain ages with the median
# Median is more resistant to very old domains than the mean

domain_age_median = (
    df_clean["Domain age years"]
    .median()
)

df_clean["Domain age years"] = (
    df_clean["Domain age years"]
    .fillna(domain_age_median)
)


# Remove the original date after creating the new variables

df_clean = df_clean.drop(
    columns=["Domain registration date"]
)


# Remove SSL expiration date
# It represents the certificate's expiration, not the domain's age,
# and its meaning depends on repeated certificate renewals.

df_clean = df_clean.drop(
    columns=["SSL certificate expire date"]
)


# %%
# 11. Review preprocessing results

print("\n==============================")
print("Preprocessed Dataset")
print("==============================")

print("\nEncoded target variable:")
print(df_clean["Label"].value_counts())

print("\nMedian domain age used for imputation:")
print(round(domain_age_median, 2))

print("\nDomain age summary:")
print(df_clean["Domain age years"].describe())

print("\nDomain registration availability indicator:")
print(
    df_clean[
        "Domain registration unavailable"
    ].value_counts()
)

print("\nFirst five rows:")
print(df_clean.head())

print("\nDataset dimensions:")
print(df_clean.shape)

print("\nMissing values after preprocessing:")
print(df_clean.isnull().sum())

print("\nRemaining data types:")
df_clean.info()

print("\nRemaining categorical variables:")
print(
    df_clean
    .select_dtypes(include=["object", "string"])
    .columns
    .tolist()
)

# %%
# 12. One-Hot Encoding

print("\n==============================")
print("One-Hot Encoding")
print("==============================")

categorical_features = [
    "SSL certificate issuer",
    "Issuer organization"
]

df_encoded = pd.get_dummies(
    df_clean,
    columns=categorical_features,
    drop_first=False,
    dtype=int
)

print("\nOriginal dataset dimensions:")
print(df_clean.shape)

print("\nEncoded dataset dimensions:")
print(df_encoded.shape)

print("\nRemaining categorical variables:")
print(
    df_encoded
    .select_dtypes(include=["object", "string"])
    .columns
    .tolist()
)

print("\nFirst five rows of the encoded dataset:")
print(df_encoded.head())

print("\nData types after encoding:")
print(df_encoded.dtypes.value_counts())
# %%

# %%
# ==========================================================
# 13. Dataset Quality Check
# ==========================================================

print("\n==============================")
print("Dataset Quality Check")
print("==============================")

# Check constant columns
constant_columns = [
    col for col in df_encoded.columns
    if df_encoded[col].nunique() == 1
]

print("\nConstant columns:")
print(constant_columns)

print("\nNumber of constant columns:")
print(len(constant_columns))


# Check duplicated columns

duplicated_columns = []

columns = df_encoded.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        if df_encoded[columns[i]].equals(df_encoded[columns[j]]):
            duplicated_columns.append(
                (columns[i], columns[j])
            )

print("\nDuplicated columns:")
print(duplicated_columns)

print("\nNumber of duplicated columns:")
print(len(duplicated_columns))


# Dataset dimensions

print("\nFinal dataset dimensions:")
print(df_encoded.shape)
# %%
# %%
# ==========================================================
# 14. Remove duplicated columns
# ==========================================================

print("\n==============================")
print("Removing Duplicated Columns")
print("==============================")

columns_to_remove = []

for col1, col2 in duplicated_columns:
    columns_to_remove.append(col2)

df_encoded = df_encoded.drop(columns=columns_to_remove)

print("\nColumns removed:")

for col in columns_to_remove:
    print("-", col)

print("\nNew dataset dimensions:")
print(df_encoded.shape)

# %%
# ==========================================================
# 15. Correlation Analysis
# ==========================================================

print("\n==============================")
print("Correlation Analysis")
print("==============================")

# Compute correlation matrix
correlation_matrix = df_encoded.corr(numeric_only=True)

# Threshold for high correlation
threshold = 0.90

high_correlations = []

columns = correlation_matrix.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        corr_value = correlation_matrix.iloc[i, j]

        if abs(corr_value) >= threshold:

            high_correlations.append(
                (
                    columns[i],
                    columns[j],
                    round(corr_value, 3)
                )
            )

print("\nHighly correlated variables (|r| >= 0.90):")

if len(high_correlations) == 0:

    print("No highly correlated variables found.")

else:

    for pair in high_correlations:

        print(pair)

print("\nNumber of highly correlated pairs:")
print(len(high_correlations))

# %%
# Correlation Heatmap

plt.figure(figsize=(12,10))

plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "/Users/catalinabuitragotorres/Documents/"
    "Potomac University/Artificial Intelligence/week 5/"
    "correlation_matrix.png",
    dpi=300
)

plt.close()

print("\nCorrelation matrix saved.")

# %%
# ==========================================================
# 16. Remove Highly Redundant Variables
# ==========================================================

print("\n==============================")
print("Removing Highly Redundant Variables")
print("==============================")

columns_to_remove = [
    "Presence of TrustPilot reviews",
    "Presence in the standard Tranco list"
]

df_encoded = df_encoded.drop(
    columns=columns_to_remove
)

print("\nRemoved variables:")

for col in columns_to_remove:
    print("-", col)

print("\nNew dataset dimensions:")
print(df_encoded.shape)

# %%
# ==========================================================
# 17. Split Features and Target
# ==========================================================

print("\n==============================")
print("Features and Target")
print("==============================")

# Separate predictors and target

X = df_encoded.drop(columns=["Label"])

y = df_encoded["Label"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nTarget distribution:")
print(y.value_counts())

# %%
# ==========================================================
# 18. Train-Test Split
# ==========================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining observations:")
print(X_train.shape)

print("\nTesting observations:")
print(X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True))

# %%
# ==========================================================
# 19. Baseline Model - Dummy Classifier
# ==========================================================

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("\n==============================")
print("Baseline Model")
print("==============================")

# Create the baseline classifier
baseline_model = DummyClassifier(
    strategy="most_frequent"
)

# Train the model
baseline_model.fit(X_train, y_train)

# Generate predictions
baseline_predictions = baseline_model.predict(X_test)

# Evaluation metrics
baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_precision = precision_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_predictions
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions
)

print("\nBaseline Performance")

print(f"Accuracy : {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall   : {baseline_recall:.4f}")
print(f"F1-score : {baseline_f1:.4f}")

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        baseline_predictions
    )
)

# %%
# ==========================================================
# 20. Decision Tree Classifier
# ==========================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("\n==============================")
print("Decision Tree Classifier")
print("==============================")

# Create the model
decision_tree = DecisionTreeClassifier(
    random_state=42
)

# Train the model
decision_tree.fit(X_train, y_train)

# Predictions
dt_predictions = decision_tree.predict(X_test)

# Evaluation metrics
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions)

print("\nDecision Tree Performance")

print(f"Accuracy : {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall   : {dt_recall:.4f}")
print(f"F1-score : {dt_f1:.4f}")

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        dt_predictions
    )
)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        dt_predictions
    )
)

# %%
# ==========================================================
# 21.Decision Tree Characteristics
# ==========================================================

print("\n==============================")
print("Decision Tree Characteristics")
print("==============================")

print("Tree depth:")
print(decision_tree.get_depth())

print("\nNumber of leaves:")
print(decision_tree.get_n_leaves())

print("\nNumber of features:")
print(decision_tree.n_features_in_)

# %%
# ==========================================================
# 22.Decision Tree Visualization
# ==========================================================

from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))

plot_tree(
    decision_tree,
    filled=True,
    feature_names=X.columns,
    class_names=["Legitimate","Fraudulent"],
    max_depth=3,
    fontsize=8
)

plt.tight_layout()

tree_path = (
    "/Users/catalinabuitragotorres/Documents/"
    "Potomac University/Artificial Intelligence/week 5/"
    "decision_tree.png"
)

plt.savefig(
    tree_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nDecision Tree saved at:")
print(tree_path)

# %%
# ==========================================================
# 23.Decision Tree Feature Importance
# ==========================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": decision_tree.feature_importances_
})

importance = (
    importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print("\nTop 15 Most Important Features")

print(
    importance.head(15)
)

# %%
# ==========================================================
# 21. Random Forest Classifier
# ==========================================================

from sklearn.ensemble import RandomForestClassifier

print("\n==============================")
print("Random Forest Classifier")
print("==============================")

# Create the model
random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train
random_forest.fit(X_train, y_train)

# Predictions
rf_predictions = random_forest.predict(X_test)

# Metrics
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

print("\nRandom Forest Performance")

print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"F1-score : {rf_f1:.4f}")

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        rf_predictions
    )
)

# %%
# ==========================================================
# Random Forest Feature Importance
# ==========================================================

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

rf_importance = (
    rf_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print("\nTop 20 Features")

print(rf_importance.head(20))