import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
from typing import List, Dict

load_dotenv()

class PubMedScraper:
    def __init__(self, api_key: str = None):
        """
        Initializes the PubMed scraper. 
        Note: If api_key is provided, rate limits are 10 req/sec instead of 3 req/sec.
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")

    def fetch(self, query: str, max_results: int = 5) -> List[Dict]:
        """Step 1 & 2: Fetch article IDs and extract raw text from PMC."""
        # 1. Search for IDs
        search_url = f"{self.base_url}esearch.fcgi"
        params = {
            "db": "pmc",
            "term": query,
            "retmode": "json",
            "retmax": max_results
        }
        if self.api_key: params["api_key"] = self.api_key
            
        print(f"[*] Fetching PMC articles for query: {query}")
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        article_ids = response.json().get("esearchresult", {}).get("idlist", [])

        articles = []
        for pmcoid in article_ids:
            # 2. Fetch full text for each ID
            fetch_url = f"{self.base_url}efetch.fcgi"
            f_params = {"db": "pmc", "id": pmcoid, "retmode": "xml"}
            if self.api_key: f_params["api_key"] = self.api_key
                
            f_resp = requests.get(fetch_url, params=f_params)
            if f_resp.status_code == 200:
                articles.append({
                    "id": f"PMC{pmcoid}",
                    "xml_content": f_resp.text,
                    "source": "PubMed Central"
                })
        return articles

    def clean(self, raw_html: str) -> str:
        """Step 3: Clean and structure the medical article."""
        soup = BeautifulSoup(raw_html, "xml")
        
        # Extract abstract and body paragraphs natively
        abstract = soup.find('abstract')
        body = soup.find('body')
        
        abstract_text = abstract.get_text(separator=' ', strip=True) if abstract else ""
        body_text = body.get_text(separator=' ', strip=True) if body else ""
        
        combined_text = f"ABSTRACT:\n{abstract_text}\n\nBODY:\n{body_text}"
        
        # Regex to clear excess whitespace / hidden HTML tags
        cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()
        return cleaned_text

    def run_pipeline(self, query: str, max_results: int = 5) -> List[Dict]:
        """Runs fetch -> clean -> summarize workflow."""
        raw_articles = self.fetch(query, max_results=max_results)
        processed_data = []
        
        for article in raw_articles:
            cleaned_text = self.clean(article["xml_content"])
            # Summarization step would link to an LLM context summarizer here
            # For ingestion, we take the raw cleaned text directly.
            processed_data.append({
                "id": article["id"],
                "content": cleaned_text,
                "source": article["source"]
            })
            
        return processed_data
