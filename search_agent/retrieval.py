import requests
import base64
import json
import time
import sys
import os
import random
from dotenv import load_dotenv
from collections import deque
import threading

MAX_RETRIES = 20
RETRY_DELAY = 2

clueweb_time_log = "./clueweb_time_log.txt"
clueweb_error_log = "./clueweb_error_log.txt"

serper_time_log = "./serper_time_log.txt"
serper_error_log = "./serper_error_log.txt"
medcorp_time_log = "./medcorp_time_log.txt"
medcorp_error_log = "./medcorp_error_log.txt"

# load_dotenv(os.environ["KEYS_ENV_PATH"], override=False)

class RateLimiter:
    """
    Simple in-process sliding window rate limiting: at most max_calls calls in any 1-second window.
    Thread-safe; suitable for single-process rate limiting in multi-threaded environments.
    """
    def __init__(self, max_calls: int, per_seconds: float = 1.0):
        self.max_calls = max_calls
        self.per = per_seconds
        self.calls = deque()  # store timestamps
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                # clean up calls outside the window
                while self.calls and now - self.calls[0] >= self.per:
                    self.calls.popleft()

                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return  

                # otherwise, wait for the oldest call to expire
                sleep_for = self.per - (now - self.calls[0])

            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                time.sleep(0.001)

# Global rate limiter: 100 QPS
SERPER_RATE_LIMITER = RateLimiter(max_calls=20, per_seconds=1.0)

def query_clueweb(query, num_docs=3):
    """
    Args:
        - query, the query to search
        - num_docs, the number of documents to return
    Returns:
        - returned_cleaned_text: a list of cleaned text strings
    """
    start_time = time.time()
    num_docs = str(num_docs)
    URL = "https://clueweb22.us"
    request_url = f"{URL}/search?query={query}&k={num_docs}"

    for attempt in range(MAX_RETRIES):
        try:
            headers = {
                "X-API-Key": os.getenv("CLUEWEB_API_KEY")
            }
            response = requests.get(request_url, headers=headers, timeout=30)
            response.raise_for_status() 
            
            json_data = response.json()
            results = json_data.get("results", [])
            return_cleaned_text = []

            for returned_document in results:
                decoded_result = base64.b64decode(returned_document).decode("utf-8")
                parsed_result = json.loads(decoded_result)

                url = parsed_result["URL"].strip()
                url_hash = parsed_result["URL-hash"]
                cweb_id = parsed_result["ClueWeb22-ID"]
                text = parsed_result["Clean-Text"]
                return_cleaned_text.append(text)
                
            end_time = time.time()
            with open(clueweb_time_log, "a") as f:
                f.write(f"query time:{end_time - start_time}\n")

            return return_cleaned_text
            
        except Exception as e:
            with open(clueweb_error_log, "a") as f:
                f.write(f"Clueweb Attempt {attempt + 1}/{MAX_RETRIES} failed, query: {query}, error: {e}\n")
            if attempt < MAX_RETRIES - 1:
                time.sleep(random.uniform(0.5, 2))
            else:
                with open(clueweb_error_log, "a") as f:
                    f.write(f"All {MAX_RETRIES} Clueweb attempts failed. Final error: {e}\n")
                raise e
            
def query_medcorp(query, num_docs=3):
    """Query the explicitly configured MedCorp service; protocol failures are fatal."""
    from urllib.parse import urlsplit
    url=os.environ.get('FAITHMED_MEDCORP_URL','').rstrip('/')
    parsed=urlsplit(url)
    if (parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username
        or parsed.password or parsed.query or parsed.fragment
        or any(marker in url.lower() for marker in ('your-host','placeholder','.example'))):
        raise ValueError('FAITHMED_MEDCORP_URL must be a real credential-free retrieval base URL')
    if not isinstance(query,str) or not query.strip() or type(num_docs) is not int or num_docs < 1:
        raise ValueError('nonempty query and positive integer document count are required')
    start_time=time.time()
    for attempt in range(MAX_RETRIES):
        try:
            response=requests.get(url+'/search',params={'query':query,'k':num_docs},timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            status=getattr(getattr(exc,'response',None),'status_code',None)
            transient=status is None or status==429 or 500 <= status < 600
            if not transient or attempt==MAX_RETRIES-1:
                raise
            time.sleep(random.uniform(.5,2))
            continue
        # Parsing is outside the network retry block. Missing fields are not empty results.
        body=response.json()
        if not isinstance(body,dict) or not isinstance(body.get('results'),list):
            raise ValueError('MedCorp response must contain a results list')
        documents=[]
        for encoded in body['results']:
            if not isinstance(encoded,str):
                raise ValueError('MedCorp result must be a base64 string')
            document=json.loads(base64.b64decode(encoded,validate=True).decode('utf-8'))
            if not isinstance(document,dict) or not isinstance(document.get('Clean-Text'),str):
                raise ValueError('MedCorp document is missing Clean-Text')
            title=document.get('title','')
            data_source=document.get('data_source','')
            if not isinstance(title,str) or not isinstance(data_source,str):
                raise ValueError('MedCorp document metadata must be strings')
            title=title.strip();data_source=data_source.strip()
            text=document['Clean-Text']
            if data_source.lower()!='pubmed' and title and text.startswith(title):
                text=text[len(title):].lstrip('. ')
            documents.append(f'Title: {title} | Data Source: {data_source}\n{text}'.strip())
        with open(medcorp_time_log,'a') as handle:
            handle.write(f'query time:{time.time()-start_time}\n')
        return documents
    raise RuntimeError('unreachable MedCorp retry state')


def query_fineweb(query, num_docs=3):
    request_url = f"https://clueweb22.us/fineweb/search?query={query}&k={num_docs}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(request_url)
            
            if response.status_code != 200:
                raise Exception(f"Error querying FineWeb: {response.status_code}")
            
            json_data = response.json()

            results = json_data.get("results", [])
            cleaned_results = []
            for returned_document in results:
                # Assuming each document in 'results' is a base64 encoded JSON string
                decoded_result = base64.b64decode(returned_document).decode("utf-8")
                parsed_result = json.loads(decoded_result)
                cleaned_results.append(parsed_result["text"])

            return cleaned_results
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(random.uniform(0.5, 2))
            else:
                raise e

def query_serper(query: str, num_docs: int = 3):
    """
    Use diskcache to cache the results, first check the cache; if not hit, trigger the real request.
    Limit the real request to ≤100 QPS.
    """
    # Ignore blank spaces in the query
    query = query.strip()
    if not query:
        return [f"No results found for blank query."]
    num_docs = max(1, int(num_docs))

    url = 'https://google.serper.dev/search'
    headers = {
        'X-API-KEY': os.getenv("SERPER_API_KEY"),
        'Content-Type': 'application/json',
    }
    data = {
        "q": query,
        "num": num_docs,
        "extendParams": {
            "country": "en",
            "page": 1,
        },
    }

    response = None
    results = None
    start_time = time.time()

    for i in range(5):
        try:
            SERPER_RATE_LIMITER.acquire()

            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code} - {response.text}")

            results = response.json()
            break
        except Exception as e:
            if i == 4:
                with open(serper_error_log, "a") as f:
                    f.write(f"Serper attempt {i + 1}/5 failed, query: {query}, error: {repr(e)}\n")
                return [f"Google search Timeout/Error, return None, Please try again later."]
            # Simple backoff
            time.sleep(min(1.0, 0.2 * (i + 1)))

    # Request was successful or we got the JSON
    try:
        if "organic" not in results:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

        web_snippets = list()
        idx = 0
        for page in results["organic"][:num_docs]:
            idx += 1
            date_published = ""
            if "date" in page:
                date_published = "\nDate published: " + page["date"]

            source = ""
            if "source" in page:
                source = "\nSource: " + page["source"]

            snippet = ""
            if "snippet" in page:
                snippet = "\n" + page["snippet"]

            redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"

            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)

        content = f"A search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        payload = [content]

        end_time = time.time()
        # randomly log 1/10 queries
        if random.random() < 0.1:
            with open(serper_time_log, "a") as f:
                f.write(f"query time:{end_time - start_time}\n")

        return payload

    except Exception as e:
        with open(serper_error_log, "a") as f:
            f.write(f"Serper parse error, query: {query}, error: {repr(e)}\n")
        return [f"No results found for '{query}'. Try with a more general query, or remove the year filter."]

def serper_credit():
    API_KEY = os.getenv("SERPER_API_KEY")

    url = "https://google.serper.dev/account"  
    headers = {
        "X-API-KEY": API_KEY
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f"Request failed: {response.status_code}, {response.text}")

if __name__ == '__main__':
    query = "post amputation respiratory distress management"
    texts = query_medcorp(query, num_docs=10)
    info_retrieved = "\n\n".join(texts)
    print(info_retrieved)
    
