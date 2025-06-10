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
