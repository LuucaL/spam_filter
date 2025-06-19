# Spam Filter Project

This project builds a spam filter using the SpamAssassin dataset. It follows a clear workflow:

- **Data Extraction:** Unpack the *.tar.bz2 archives into a centralized directory.
- **Preprocessing:** Parse the email files and clean their content by removing headers, converting text to lowercase, and replacing URLs and numbers with placeholders.
- **Model Training:** Train a machine learning pipeline that uses vectorization, TF-IDF transformation, and a Linear SVC. Hyperparameters are tuned using GridSearchCV.
- **Evaluation:** Evaluate the model with classification reports and confusion matrices.

## Project Structure

````markdown

spam-filter-project/
├── data/                     
│   ├── 20021010_easy_ham.tar.bz2
│   ├── …
├── notebooks/
│   └── SpamFilter.ipynb      
├── spam_filter.py            
└── requirements.txt          
````


# Function Comments for spam_filter.py

---

### extract_archive

This function extracts the contents of a .tar.bz2 archive into the specified directory.
It ensures that all files from the archive are unpacked and available for further processing.

---

### parse_email

This function parses an email file, removes its headers, and returns the cleaned body text.
It is used to prepare raw email data for further preprocessing steps.

---

### clean_text

This function cleans the email text by converting it to lowercase, removing special characters,
and replacing URLs and numbers with placeholder tokens. It standardizes the text for model input.

---

### load_emails_from_dir

This function loads all emails from a directory, parses and cleans them, and returns a list of processed emails.
It automates the data loading and preprocessing pipeline for the dataset.

---

### build_pipeline

This function builds and returns a machine learning pipeline consisting of a vectorizer,
TF-IDF transformer, and a Linear SVC classifier. It encapsulates the model architecture.

---

### tune_hyperparameters

This function performs hyperparameter tuning using GridSearchCV on the provided pipeline and data.
It returns the best estimator found during the search.

---

### evaluate_model

This function evaluates the trained model on test data, printing classification reports and confusion matrices.
It provides insights into the model's performance.
