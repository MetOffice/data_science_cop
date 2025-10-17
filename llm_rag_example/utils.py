from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse


def get_all_links(start_url, max_pages=1500):
    # Crawl a site and return internal links (up to max_pages).
    visited = set()
    to_visit = [start_url]
    domain = urlparse(start_url).netloc
    all_links = []
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            res = requests.get(url, timeout=10)
            visited.add(url)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                # Stay within same domain
                if urlparse(link).netloc == domain and link not in visited:
                    to_visit.append(link)
                    all_links.append(link)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    return list(set(all_links))

def load_website(url, max_pages=3000):
    # Load all internal pages as LangChain documents.
    links = get_all_links(url, max_pages=max_pages)
    print(f"Found {len(links)} pages to load...")
    try:
        loader = WebBaseLoader(links)
        data = loader.load()
    except Exception as e:
        print(f"Error loading data: {e}")
    return data

def clean_text(text):
    # Strip HTML tags (before regex cleaning)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list:
    # Splits long text into overlapping chunks suitable for LLMs.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks
