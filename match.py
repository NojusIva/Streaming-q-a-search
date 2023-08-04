from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np


def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = [line.rstrip() for line in f]
        return lines


def find_matching_questions(sentence, top_n=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    data = read_file('questions.txt')
    qna_embeddings = model.encode(data)
    sentence_embedding = model.encode([sentence])
    similarities = cosine_similarity(sentence_embedding, qna_embeddings).flatten()
    most_similar_indices = np.argsort(similarities)[-top_n:]

    return [data[i] for i in most_similar_indices[::-1]]


def process_conversation_segments(file_name):
    with open(file_name, 'r') as f:
        for line in f:
            data = line.rstrip()
            sentences = sent_tokenize(data)
            for i in range(len(sentences)):
                context = ' '.join(sentences[max(0, i-2):i+1])
                matching_questions = find_matching_questions(context)
                print(f'Input: {context}\nMatched Questions: {matching_questions}\n')


def main():
    process_conversation_segments('transcript2.txt')


if __name__ == "__main__":
    main()

