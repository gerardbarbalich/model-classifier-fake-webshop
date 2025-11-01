"""
This script produces a classsifier of fake webshops. 
To do this, it injests processed data, trains and tests a classifer, and outputs the pickled classifier. 

The classifier is a Pipeline() containing a TfidfVectorizer() and a RandomForestClassifier()

In more detail - it imports data from WebScan, combines it, and performs analysis to identify fake webshops. The expected
inputs include data files in pickle format and a CSV file containing confirmed fake webshops. It generates
a model pipeline using RandomForestClassifier to classify webshops as 'Yes' (fake) or 'No' (not fake) based
on textual and domain features.

Expected Inputs:
- Pickle files with webshop data
- CSV file ('df_fake_domains.csv') containing confirmed fake webshops

Expected Output:
- Trained pipeline ('fake-webshop-pipeline-[date].pkl')
- Evaluation metrics such as confusion matrix, precision, recall, accuracy, F1 score

Dependencies:
- pandas, nltk, scikit-learn
"""

import os
import glob
import pandas as pd
import joblib
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Any

# Configuration dictionary to store all data, models, and settings
data: Dict[str, Any] = {
    'DATA_DIR': str,
    'SEL_DATE': '2018-12-14',      # Date of the WebScan data to use
    'SPLIT_SEED': 2,                # Random seed for reproducibility
    'SPLIT_TEST_SIZE': 0.5,         # 50% of data reserved for testing
    'clf': RandomForestClassifier,
    'd': pd.DataFrame,
    'data_files': str,
    'df': pd.DataFrame,
    'df_fake_domains': pd.DataFrame,
    'dfs': list,
    'ls_custom_stop': list,
    'ls_custom_stop_set': list,
    'ls_fake_domains': list,
    'tokenizer': RegexpTokenizer,
    'vec_tfidf': TfidfVectorizer
}

# Define the directory containing WebScan pickle files
# Expected to be in user's home directory with format: dns-flag-day-YYYY-MM-DD-content
data['DATA_DIR'] = os.path.join(os.environ['HOME'], f"dns-flag-day-{data['SEL_DATE']}-content")

# ============================================================================
# SECTION 1: Import and Combine WebScan Data
# ============================================================================

# Load all pickle files from the WebScan data directory
data['data_files'] = glob.glob(os.path.join(data['DATA_DIR'], '*.p'))
print(f"Found {len(data['data_files'])} data files")

# Read each pickle file into a DataFrame and combine them
data['dfs'] = [pd.read_pickle(file) for file in data['data_files']]
data['d'] = pd.concat(data['dfs'], ignore_index=True)
print(f"Combined dataset has {len(data['d'])} rows")

# ============================================================================
# SECTION 2: Import Known Fake Domains
# ============================================================================

# Load the list of confirmed fake webshop domains for labeling
data['df_fake_domains'] = pd.read_csv('df_fake_domains.csv')
data['ls_fake_domains'] = data['df_fake_domains']['domains'].values.tolist()
print(f"Loaded {len(data['ls_fake_domains'])} confirmed fake domains")

# ============================================================================
# SECTION 3: Aggregate and Label Data
# ============================================================================

# Aggregate all body text for each domain into a single text corpus
# This combines multiple pages from the same domain into one training sample
data['df'] = (data['d'][['domain', 'BodyText']]
              .groupby('domain')['BodyText']
              .apply(lambda x: " ".join(x))
              .reset_index())

# Create binary labels: 'Yes' for confirmed fakes, 'No' for legitimate
data['df']['label'] = data['df']['domain'].apply(
    lambda x: 'Yes' if x in data['ls_fake_domains'] else 'No'
)

print(f"Total domains: {len(data['df'])}")
print(f"Fake domains: {(data['df']['label'] == 'Yes').sum()}")
print(f"Legitimate domains: {(data['df']['label'] == 'No').sum()}")

# ============================================================================
# SECTION 4: Configure Text Processing
# ============================================================================

# Define custom stop words list
# Includes standard English stop words plus domain-specific terms that don't help classification
data['ls_custom_stop'] = stopwords.words('english') + [
    'nz', 'st', 'www', 'co', 'new', 'zealand', 'nzd', 'us', 'ml', 'javascript'
]
data['ls_custom_stop_set'] = set(data['ls_custom_stop'])

# Configure tokenizer to extract words (2+ characters, letters only)
data['tokenizer'] = RegexpTokenizer(r'[a-zA-Z]{2,}')


def gen_tokens(text: str) -> List[str]:
    """
    Tokenize and clean text for TF-IDF vectorization.
    
    This function:
    1. Tokenizes text into words (2+ letters)
    2. Converts all words to lowercase
    3. Filters out stop words
    
    Args:
        text: Raw text string to tokenize
    
    Returns:
        List of cleaned, lowercase tokens
    """
    return [
        w.lower() 
        for w in data['tokenizer'].tokenize(text) 
        if w.lower() not in data['ls_custom_stop_set']
    ]

# ============================================================================
# SECTION 5: Build Classification Pipeline
# ============================================================================

# Build a scikit-learn pipeline combining feature extraction and classification
# The pipeline ensures consistent preprocessing during training and prediction

# Step 1: TF-IDF Vectorization
# Converts text into numerical features based on term frequency-inverse document frequency
vectorizer = TfidfVectorizer(
    stop_words=data['ls_custom_stop'],  # Remove common words
    tokenizer=gen_tokens,                # Use custom tokenizer
    max_features=1000,                   # Limit to top 1000 features to avoid overfitting
    ngram_range=(1, 2)                   # Use unigrams and bigrams (1-2 word phrases)
)

# Step 2: Random Forest Classifier
# Ensemble method that builds multiple decision trees and averages their predictions
classifier = RandomForestClassifier(
    random_state=42,        # Seed for reproducibility
    max_features='auto',    # Consider sqrt(n_features) at each split
    n_estimators=200        # Build 200 decision trees
)

# Combine steps into a single pipeline
data['clf'] = Pipeline([
    ('tfidf', vectorizer),
    ('classifier', classifier)
])

print("Pipeline constructed successfully")

# ============================================================================
# SECTION 6: Split Data and Train Model
# ============================================================================

# Split data into training and testing sets
# Using 50/50 split to balance training data and test reliability
x_train, x_test, y_train, y_test = train_test_split(
    data['df']['BodyText'],
    data['df']['label'],
    test_size=data['SPLIT_TEST_SIZE'],
    random_state=data['SPLIT_SEED'],
    stratify=data['df']['label']  # Maintain class balance in splits
)

print(f"\nTraining set size: {len(x_train)}")
print(f"Test set size: {len(x_test)}")

# Train the pipeline on the training data
print("\nTraining model...")
data['clf'].fit(x_train, y_train)
print("Training complete")

# ============================================================================
# SECTION 7: Evaluate Model Performance
# ============================================================================

# Generate predictions on the test set
print("\nGenerating predictions on test set...")
y_pred = data['clf'].predict(x_test)

# Calculate confusion matrix
# Format: [[TN, FP], [FN, TP]]
cnf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cnf_matrix.ravel()

# Display evaluation metrics
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)

print("\nConfusion Matrix:")
print(cnf_matrix)
print(f"True Negatives: {tn}, False Positives: {fp}")
print(f"False Negatives: {fn}, True Positives: {tp}")

print("\nPerformance Metrics:")
print("-" * 60)

# Recall: Of all actual fakes, what percentage did we correctly identify?
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"Recall (Sensitivity):  {recall:.3f}")
print(f"  → {recall*100:.1f}% of fake webshops correctly identified")

# Precision: Of all predicted fakes, what percentage were actually fake?
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
print(f"\nPrecision:             {precision:.3f}")
print(f"  → {precision*100:.1f}% of flagged sites were actually fake")

# Accuracy: What percentage of all predictions were correct?
accuracy = (tp + tn) / (tn + fn + tp + fp)
print(f"\nAccuracy:              {accuracy:.3f}")
print(f"  → {accuracy*100:.1f}% of all predictions were correct")

# F1 Score: Harmonic mean of precision and recall
f1 = f1_score(y_test, y_pred, average="macro")
print(f"\nF1 Score:              {f1:.3f}")
print(f"  → Overall model quality (balance of precision and recall)")

# ============================================================================
# SECTION 8: Cross-Validation
# ============================================================================

# Perform 10-fold cross-validation on training data to check for overfitting
print("\n" + "="*60)
print("CROSS-VALIDATION (10-Fold)")
print("="*60)
scores = cross_val_score(data['clf'], x_train, y_train, cv=10, scoring='accuracy')
print(f"Individual fold scores: {scores}")
print(f"Mean CV accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# ============================================================================
# SECTION 9: Export Trained Pipeline
# ============================================================================

# Generate filename with current date for versioning
current_date = datetime.now().strftime('%Y-%m-%d')
output_filename = f'fake-webshop-pipeline-{current_date}.pkl'

# Save the trained pipeline
joblib.dump(data['clf'], output_filename)
print(f"\n{'='*60}")
print(f"Model saved to: {output_filename}")
print(f"{'='*60}")
