# Fake Webshop Classifier

A machine learning pipeline that classifies websites as legitimate or fake (fraudulent) based on their textual content. This project uses Natural Language Processing (NLP) and Random Forest classification to identify suspicious webshops from web scraping data.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Testing](#testing)

## Overview

This classifier helps identify fake or fraudulent webshops by analyzing their website content. The system:

1. Processes text scraped from websites (WebScan data)
2. Extracts TF-IDF features from the text content
3. Trains a Random Forest classifier to distinguish legitimate from fake sites
4. Exports a ready-to-use classification pipeline

**Key Features:**
- Text preprocessing with custom tokenization and stop word removal
- TF-IDF vectorization with unigram and bigram features
- Random Forest ensemble classification (200 trees)
- Comprehensive evaluation metrics and cross-validation
- Exportable pipeline for production use

## Installation

### Prerequisites
- Python 3.8 or higher
- WebScan data files (pickle format)
- List of confirmed fake domains (CSV file)

### Setup

1. Navigate to the project directory:
```bash
cd model-classifier-fake-webshop
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK stopwords:
```python
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

### Preparing Your Data

1. **WebScan Data**: Place your WebScan pickle files in a directory with this format:
   ```
   ~/dns-flag-day-YYYY-MM-DD-content/*.p
   ```

2. **Fake Domains List**: Create a CSV file named `df_fake_domains.csv` with a column named `domains`:
   ```csv
   domains
   fake-shop1.com
   scam-site2.com
   fraudulent-store3.com
   ```

### Training the Model

Run the training script:
```bash
python train-test-export-fake-webshop-classifier.py
```

The script will:
- Load and combine all WebScan data files
- Process and tokenize the text
- Train the classification pipeline
- Display evaluation metrics
- Save the trained model with today's date

### Example Output

```
Found 45 data files
Combined dataset has 12543 rows
Loaded 328 confirmed fake domains
Total domains: 8912
Fake domains: 328
Legitimate domains: 8584

Training model...
Training complete

============================================================
MODEL EVALUATION RESULTS
============================================================

Confusion Matrix:
[[4201   91]
 [  42  122]]

Performance Metrics:
------------------------------------------------------------
Recall (Sensitivity):  0.744
  → 74.4% of fake webshops correctly identified

Precision:             0.573
  → 57.3% of flagged sites were actually fake

Accuracy:              0.970
  → 97.0% of all predictions were correct

F1 Score:              0.822
  → Overall model quality (balance of precision and recall)
```

## Data Requirements

### Input Files

1. **WebScan Pickle Files** (`*.p`)
   - Must contain columns: `domain`, `BodyText`
   - Each file represents scraped content from websites
   - Located in: `~/dns-flag-day-YYYY-MM-DD-content/`

2. **Fake Domains CSV** (`df_fake_domains.csv`)
   - Single column named `domains`
   - Contains confirmed fake/fraudulent domain names
   - Used for labeling training data

### Expected Format

WebScan data structure:
```python
{
    'domain': 'example-shop.com',
    'BodyText': 'Welcome to our online store...'
}
```

Fake domains CSV:
```csv
domains
fake-shop1.com
scam-site2.com
```

## How It Works

### Pipeline Architecture

```
Raw Text → Tokenization → TF-IDF Vectorization → Random Forest → Prediction
```

### Step-by-Step Process

1. **Data Loading**
   - Reads multiple WebScan pickle files
   - Combines them into a single DataFrame
   - Imports known fake domain list for labeling

2. **Text Aggregation**
   - Groups all text content by domain
   - Creates one training sample per domain
   - Assigns binary labels ('Yes'=fake, 'No'=legitimate)

3. **Text Preprocessing**
   - Tokenizes text using regex (2+ letter words)
   - Removes stop words (English + custom list)
   - Converts to lowercase
   - Filters domain-specific noise terms

4. **Feature Extraction**
   - TF-IDF vectorization with top 1000 features
   - Includes unigrams and bigrams
   - Captures word importance across documents

5. **Model Training**
   - Random Forest with 200 trees
   - 50/50 train-test split
   - Stratified sampling to maintain class balance
   - 10-fold cross-validation

6. **Evaluation & Export**
   - Calculates confusion matrix and key metrics
   - Performs cross-validation
   - Exports pipeline as pickle file

### Custom Stop Words

The classifier filters out common words that don't help classification:
- Standard English stop words (from NLTK)
- Domain-specific terms: `www`, `co`, `nz`, `new zealand`, etc.
- Technical terms: `javascript`, `ml`

## Model Performance

### Metrics Explained

- **Recall (Sensitivity)**: Of all actual fake sites, what % did we catch?
  - Higher recall = fewer fake sites slip through
  - Important for consumer protection

- **Precision**: Of all sites we flagged as fake, what % were actually fake?
  - Higher precision = fewer false accusations
  - Important for avoiding legitimate business harm

- **Accuracy**: What % of all predictions were correct?
  - Overall correctness measure
  - Can be misleading with imbalanced data

- **F1 Score**: Harmonic mean of precision and recall
  - Balanced measure of model quality
  - Useful when classes are imbalanced

### Expected Performance

With typical WebScan data:
- Accuracy: ~95-98%
- Recall: ~70-85%
- Precision: ~55-70%
- F1 Score: ~75-85%

Note: Performance depends heavily on data quality and the number of confirmed fake domains available for training.

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Tests cover:
- Text tokenization and preprocessing
- Feature extraction pipeline
- Model training consistency
- Prediction output format
- Edge cases (empty text, special characters)

## Configuration

Edit these variables in the script to customize:

```python
data = {
    'SEL_DATE': '2018-12-14',    # Date of WebScan data
    'SPLIT_SEED': 2,              # Random seed for reproducibility
    'SPLIT_TEST_SIZE': 0.5,       # Test set proportion (0.5 = 50%)
}
```

Adjust model hyperparameters:
```python
RandomForestClassifier(
    n_estimators=200,      # Number of trees (more = better but slower)
    max_features='auto',   # Features per split
    random_state=42        # Reproducibility seed
)

TfidfVectorizer(
    max_features=1000,     # Maximum vocabulary size
    ngram_range=(1, 2),    # Use 1 and 2-word phrases
)
```

## Troubleshooting

**"Data directory not found"**
- Check that `DATA_DIR` path matches your WebScan data location
- Verify the date format in the directory name

**"df_fake_domains.csv not found"**
- Ensure the CSV file is in the same directory as the script
- Check that the column is named `domains` (lowercase)

**Poor model performance**
- Increase the number of known fake domains in your training data
- Adjust the train-test split ratio
- Tune Random Forest hyperparameters (n_estimators, max_depth)

**NLTK stopwords error**
- Run: `python -c "import nltk; nltk.download('stopwords')"`

## Output Files

The script generates:
- `fake-webshop-pipeline-YYYY-MM-DD.pkl`: Trained classification pipeline

To use the saved model:
```python
import joblib

# Load the pipeline
pipeline = joblib.load('fake-webshop-pipeline-2024-11-01.pkl')

# Make predictions
websites = ["website text content here"]
predictions = pipeline.predict(websites)
# Returns: ['Yes'] for fake, ['No'] for legitimate
```

## License

This project is provided as-is for research and educational purposes.

