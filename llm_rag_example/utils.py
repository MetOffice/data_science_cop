from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import asyncio
import aiohttp
from typing import List
import warnings
import urllib3

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


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

def get_all_links_async(start_url, max_pages=1500):
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
            all_links.append(url)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    
    return list(set(all_links))


async def fetch_url(session: aiohttp.ClientSession, url: str) -> tuple:
    # Fetch a single URL and return (url, content).
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15), headers=headers, ssl=False) as response:
            if response.status == 200:
                content = await response.text()
                return (url, content)
            else:
                print(f"Status {response.status} for {url}")
                return (url, None)
    except asyncio.TimeoutError:
        print(f"Timeout fetching {url}")
        return (url, None)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return (url, None)

async def fetch_batch(links: List[str], max_concurrent: int = 20) -> dict:
    # Fetch a batch of URLs concurrently using aiohttp.
    connector = aiohttp.TCPConnector(
        limit=max_concurrent, 
        limit_per_host=5,
        ssl=False,
        use_dns_cache=True
    )
    timeout = aiohttp.ClientTimeout(total=30, connect=15)
    
    async with aiohttp.ClientSession(
        connector=connector, 
        timeout=timeout,
        trust_env=True
    ) as session:
        tasks = [fetch_url(session, link) for link in links]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        content_dict = {}
        for result in results:
            if isinstance(result, tuple):
                url, content = result
                if content:
                    content_dict[url] = content
            else:
                print(f"Error in fetch_batch: {result}")
        
        return content_dict

def parse_and_load_documents(content_dict: dict) -> List:
    # Parse HTML content and create LangChain documents.
    documents = []
    
    for url, content in content_dict.items():
        try:
            soup = BeautifulSoup(content, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator="\n", strip=True)
            
            # Create a simple document structure similar to WebBaseLoader
            from langchain_core.documents import Document
            doc = Document(page_content=text, metadata={"source": url})
            documents.append(doc)
        except Exception as e:
            print(f"Error parsing {url}: {e}")
    
    return documents

async def load_website_async(url, max_pages=3000, batch_size=50, max_concurrent=20):
    # Load all internal pages as LangChain documents using asyncio + aiohttp.
   
    links = get_all_links_async(url, max_pages=max_pages)
    print(f"Found {len(links)} pages to load...")
    
    all_documents = []
    total_batches = (len(links) + batch_size - 1) // batch_size
    
    # Process links in batches
    for batch_idx, batch_start in enumerate(range(0, len(links), batch_size)):
        batch_end = min(batch_start + batch_size, len(links))
        batch_links = links[batch_start:batch_end]
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} "
              f"({len(batch_links)} URLs, max {max_concurrent} concurrent)...")
        
        try:
            # Fetch all URLs in batch concurrently
            content_dict = await fetch_batch(batch_links, max_concurrent=max_concurrent)
            
            # Parse and convert to documents
            documents = parse_and_load_documents(content_dict)
            all_documents.extend(documents)
            
            print(f"Batch {batch_idx + 1}: Loaded {len(documents)} documents")
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
        
        # Optional: Small delay between batches
        if batch_end < len(links):
            await asyncio.sleep(0.5)
    
    print(f"Successfully loaded {len(all_documents)} documents")
    return all_documents

def load_website_in_batch(url, max_pages=3000, batch_size=50, max_concurrent=20):
    # Synchronous wrapper for async function.
    return asyncio.run(load_website_async(url, max_pages, batch_size, max_concurrent))