# FAQ Retrieval System

This project is a basic FAQ retrieval system that searches for the best-matched answers to user queries.

## Requirements
- Python 3.6+
- pandas
- scikit-learn

## Setup Instructions
1. Clone this repository.

2. Install the required libraries:
pip install pandas scikit-learn

3. Ensure the dataset is properly formatted:
- The dataset should be a CSV file named faq_system_dataset.csv in the same directory as the script.
- The file must use a semicolon (;) as the delimiter between the question and answer columns.
- The column names in your dataset Should be " Question" and "Answer"

4. Run the script:
python faq_retrieval_system.py


## Assumptions
- The dataset should be a CSV file named `faq_dataset.csv` in the same directory as the script.

## Challenges Faced
- Handling rephrased or semantically similar questions with limited data.
- Adjusting the similarity threshold for better accuracy.
