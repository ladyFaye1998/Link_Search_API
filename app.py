import os
import sys
import json
import logging
import re
from flask import Flask, request, jsonify
import numpy as np
import faiss
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
import openai

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Initialize Flask app
app = Flask(__name__)

# Initialize the OpenAI API key
# Set your OpenAI API key here
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
#openai.api_key = 'sk-XXXX'
if openai.api_key:
    logger.info("OpenAI API key loaded successfully.")
else:
    logger.error("OpenAI API key is missing.")
    sys.exit(1)  # Exit the application if the API key is essential



# Global variables for ANN index and embeddings
ann_index = None
content_embeddings = None
bm25 = None  # BM25 instance
tokenized_corpus = []  # For BM25
links = []  # Initialize links

# Load NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load stop words and initialize lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to load links from 'All_Links.json'
def load_links_from_json(file_path='All_Links.json'):
    logger.debug(f"Attempting to load links from {file_path}")
    links = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                url = item.get('url', '').strip()
                description = item.get('description', '').strip()
                if url:
                    links.append({'url': url, 'description': description})
        if not links:
            logger.warning(f"No links found in '{file_path}'.")
        else:
            logger.debug(f"Loaded {len(links)} links from '{file_path}'.")
    except FileNotFoundError:
        logger.error(f"Error: '{file_path}' file not found.")
    except Exception as e:
        logger.error(f"Error reading '{file_path}': {e}", exc_info=True)
    return links


# Helper function: Text preprocessing
def preprocess_text(text):
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text, flags=re.MULTILINE)
    # Remove special characters
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    # Tokenize
    tokens = nltk.word_tokenize(text.lower())
    # Remove stop words and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Helper function: Extract keywords from URL
def extract_keywords_from_url(url):
    parsed_url = requests.utils.urlparse(url)
    path = parsed_url.path  # get the path part of the URL
    # Remove file extension
    path = re.sub(r'\.\w+$', '', path)
    # Split by '/', '-', '_', etc.
    tokens = re.split(r'[/\-_]+', path)
    # Remove empty tokens
    tokens = [token for token in tokens if token]
    # Remove stop words and non-alphabetic tokens
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return tokens

# Helper function: Fetch and process link content
def fetch_and_process_link_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        session = requests.Session()
        session.max_redirects = 5  # Limit redirects to prevent loops
        response = session.get(url, timeout=10, headers=headers, allow_redirects=True)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        # Get page title
        title = soup.title.string if soup.title else ''
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        # Extract text
        text = soup.get_text(separator=' ')
        tokens = preprocess_text(f"{title} {text}")
        if not tokens:
            logger.warning(f"No content extracted from {url}.")
        return tokens

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching content from {url}: {e}", exc_info=True)
        return []

# Fetch and preprocess content for all links
def fetch_all_contents(links):
    contents = []
    url_contents = {}
    for idx, link in enumerate(links):
        logger.info(f"Fetching content for link {idx+1}/{len(links)}: {link['url']}")
        content_tokens = fetch_and_process_link_content(link['url'])
        url_keywords = extract_keywords_from_url(link['url'])
        all_tokens = preprocess_text(link['description']) + content_tokens + url_keywords
        link['combined_tokens'] = all_tokens
        combined_text = ' '.join(all_tokens)
        contents.append(combined_text)
        tokenized_corpus.append(all_tokens)
        # Store the combined text in the link dictionary
        link['content'] = combined_text
        # Also store in url_contents for caching
        url_contents[link['url']] = combined_text
    return contents, url_contents

# Precompute contents, embeddings, BM25 index, and build ANN index
def precompute_contents(re_fetch=False, cache_file='cached_contents.json'):
    global content_embeddings, bm25, links, tokenized_corpus

    logger.info(f"Starting precompute_contents with re_fetch={re_fetch}")

    if not re_fetch and os.path.exists(cache_file):
        logger.info("Loading contents and embeddings from cache...")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            links = cached_data['links']
            tokenized_corpus = cached_data['tokenized_corpus']
            content_embeddings = np.array(cached_data['embeddings'], dtype='float32')
            if content_embeddings.size == 0:
                raise ValueError("Embeddings loaded from cache are empty.")
            logger.info("Loaded cached data successfully.")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}", exc_info=True)
            re_fetch = True  # Force re-fetching if cache is invalid

    if re_fetch:
        logger.info("Fetching and processing content from all links.")
        # Load links from JSON file
        links = load_links_from_json('All_Links.json')
        if not links:
            logger.error("No links to process. Exiting.")
            return
        else:
            logger.debug(f"Loaded {len(links)} links.")

        # Reset tokenized_corpus
        tokenized_corpus = []
        # Fetch contents and save to cache
        contents, url_contents = fetch_all_contents(links)
        logger.debug("Fetched and processed all contents.")

        # Now compute embeddings
        logger.info("Computing embeddings using OpenAI API...")
        content_embeddings = []
        batch_size = 16  # Adjust as needed
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i+batch_size]
            # Skip empty contents
            batch = [text for text in batch if text.strip()]
            if not batch:
                continue
            # Truncate content to avoid exceeding token limits
            batch = [text[:8000] for text in batch]
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                embeddings = [data_point['embedding'] for data_point in response['data']]
                content_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Error computing embedding: {e}", exc_info=True)
                # Skip embeddings for this batch
                continue

        if not content_embeddings:
            logger.error("No embeddings were computed. Cannot proceed.")
            return
        else:
            content_embeddings = np.array(content_embeddings, dtype='float32')
            # Save contents and embeddings to cache
            logger.info("Saving contents and embeddings to cache...")
            cached_data = {
                'links': links,
                'tokenized_corpus': tokenized_corpus,
                'embeddings': content_embeddings.tolist()
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f)
            logger.info("Contents and embeddings saved to cache.")

    # Build ANN index after embeddings are loaded or computed
    if content_embeddings is not None and content_embeddings.size > 0:
        build_ann_index(content_embeddings)
    else:
        logger.error("No embeddings are available. ANN index will not be built.")

    # Build BM25 index
    if tokenized_corpus:
        global bm25
        bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built.")
    else:
        logger.error("Tokenized corpus is empty. BM25 index not built.")


# Helper function: Build ANN index using FAISS
def build_ann_index(embeddings):
    global ann_index
    embeddings_np = embeddings.astype('float32')
    dimension = embeddings_np.shape[1]
    # Normalize embeddings
    faiss.normalize_L2(embeddings_np)
    ann_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    ann_index.add(embeddings_np)
    logger.info("Built ANN index.")

# Function to search for the most relevant links based on query
def find_relevant_links(query, top_n=3):
    if not links or content_embeddings is None or ann_index is None or bm25 is None:
        logger.warning("Data is not ready for searching. Returning default results. ")
        # Return top_n empty results
        return [{'url': '', 'description': '', 'score': 0.0} for _ in range(top_n)]

    # Step 1: Preprocess the query
    query_tokens = preprocess_text(query)
    query_processed = ' '.join(query_tokens)

    # Step 2: Compute query embedding using OpenAI embeddings
    try:
        response = openai.Embedding.create(
            input=[query_processed[:8000]],  # Limit to 8000 characters
            model="text-embedding-ada-002"
        )
        query_embedding = np.array(response['data'][0]['embedding'], dtype='float32').reshape(1, -1)
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
    except Exception as e:
        logger.error(f"Error computing query embedding: {e}", exc_info=True)
        return [{'url': '', 'description': '', 'score': 0.0} for _ in range(top_n)]

    # Step 3: Perform ANN search
    num_candidates = min(len(links), 100)
    distances, indices = ann_index.search(query_embedding, num_candidates)
    indices = indices[0]
    distances = distances[0]

    # Step 4: Collect candidate links
    candidate_links = []
    candidate_indices = []
    for idx in indices:
        link = links[idx]
        candidate_links.append(link)
        candidate_indices.append(idx)

    # Step 5: BM25 ranking on candidates
    candidate_tokens = [tokenized_corpus[idx] for idx in candidate_indices]
    bm25_candidate = BM25Okapi(candidate_tokens)
    bm25_candidate_scores = bm25_candidate.get_scores(query_tokens)
    # Normalize BM25 scores
    max_bm25_score = max(bm25_candidate_scores) if bm25_candidate_scores.size > 0 else 1
    normalized_bm25_scores = bm25_candidate_scores / max_bm25_score if max_bm25_score != 0 else bm25_candidate_scores

    # Step 6: Combine Scores
    combined_results = []
    for idx, (candidate, bm25_score, distance) in enumerate(zip(candidate_links, normalized_bm25_scores, distances)):
        # Cosine similarity is between 0 and 1 after normalization
        ann_score = distance
        combined_score = 0.3 * ann_score + 0.7 * bm25_score  # Adjusted weights
        candidate['combined_score'] = round(combined_score, 4)
        # Ensure 'description' is not too long
        description = candidate.get('description', '')
        if len(description) > 200:
            description = description[:197] + '...'
        combined_results.append({
            'url': candidate.get('url', ''),
            'description': description,
            'score': candidate['combined_score']
        })

    # Step 7: Re-rank using OpenAI's GPT model
    top_candidates = sorted(combined_results, key=lambda x: x['score'], reverse=True)[:top_n * 2]

    re_ranked_results = []

    for candidate in top_candidates:
        messages = [
            {
                "role": "system",
                "content": "You are to act as a numerical evaluator. Given a query and a candidate description, you must output only a single number between 0 and 1 indicating the relevance, and nothing else."
            },
            {
                "role": "user",
                "content": f"Query: {query}\nCandidate Description: {candidate['description']}\n\nPlease output only the relevance score between 0 and 1."
            }
        ]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=5,
                temperature=0
            )
            relevance_score_text = response['choices'][0]['message']['content'].strip()
            print(f"Assistant's response: {relevance_score_text}")  # For debugging
            try:
                relevance_score = float(relevance_score_text)
            except ValueError:
                match = re.findall(r"[\d\.]+", relevance_score_text)
                if match:
                    relevance_score = float(match[0])
                else:
                    relevance_score = 0.0  # Default to 0 if no number is found
            candidate['gpt_score'] = relevance_score
            re_ranked_results.append(candidate)
        except Exception as e:
            logger.error(f"Error during GPT re-ranking: {e}", exc_info=True)
            print(f"Exception occurred: {e}")
            candidate['gpt_score'] = 0.0
            re_ranked_results.append(candidate)

    # Final ranking based on GPT scores
    final_ranked_candidates = sorted(re_ranked_results, key=lambda x: x['gpt_score'], reverse=True)

    # Collect top_n links
    top_links = final_ranked_candidates[:top_n]

    return top_links

# API endpoint to search for links
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = find_relevant_links(query)
    return jsonify(results), 200

# Add a route for the root URL
@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Enhanced Link Search API!",
        "endpoints": {
            "/search": "Search for relevant links. Use '?query=' parameter with your search term."
        }
    })

if __name__ == '__main__':
    # Set re_fetch based on an environment variable
    re_fetch = os.environ.get('RE_FETCH_CONTENT', 'false').lower() == 'true'
    cache_file = 'cached_contents.json'

    # Check if cache file exists; if not, set re_fetch to True
    if not os.path.exists(cache_file):
        logger.info("Cache file not found. Fetching content and computing embeddings.")
        re_fetch = True

    # Precompute contents and build ANN index before starting the server
    precompute_contents(re_fetch)

    # Use the port provided by the environment or default to 10000
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)




