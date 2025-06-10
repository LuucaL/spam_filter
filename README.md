# Spam Filter

This project implements a spam filter using the SpamAssassin dataset. The key steps include:

- **Data Extraction:** Extracting *.tar.bz2 archives into a working directory.
- **Preprocessing:** Parsing each email, cleaning the text (removing headers, lowercasing, masking URLs and numbers).
- **Model Training:** Building a machine learning pipeline (vectorization, TF-IDF transformation, Linear SVC) with hyperparameter tuning via GridSearchCV.
- **Evaluation:** Generating classification reports and confusion matrices for assessment.



spam-filter-project/
│
├── data/                     
│   ├── 20021010_easy_ham.tar.bz2
│   ├── …
│
├── notebooks/
│   └── SpamFilter.ipynb      
│
├── spam_filter.py            
└── requirements.txt          
