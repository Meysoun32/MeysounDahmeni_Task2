#import the necessairy libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
def load_faq_data(file_path):
    return pd.read_csv(file_path)

# Preprocess and vectorize the dataset
def prepare_faq_system(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(data['Question'])
    return vectorizer, vectors

# Search functionality
def search_faq(user_query, vectorizer, vectors, data):
    query_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vector, vectors).flatten()
    best_match_index = similarities.argmax()
    if similarities[best_match_index] > 0.1:  # Threshold for relevance
        return data.iloc[best_match_index]['Answer']
    else:
        return "Sorry, I couldn't find a matching FAQ. Please try rephrasing your question."

# Main function
def main():
    # Load and prepare data
    faq_data = load_faq_data("faq_system_dataset.csv")
    tfidf_vectorizer, faq_vectors = prepare_faq_system(faq_data)

    print("Welcome to the FAQ Retrieval System!")
    while True:
        user_question = input("Ask a question (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            print("Thank you for using the FAQ Retrieval System!")
            break
        answer = search_faq(user_question, tfidf_vectorizer, faq_vectors, faq_data)
        print(f"Answer: {answer}")

if __name__ == "__main__":
     main()