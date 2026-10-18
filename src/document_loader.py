import re, os
import json
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations
from unstructured.cleaners.core import clean_extra_whitespace
from dotenv import load_dotenv, find_dotenv
from unstructured_client.models.errors import SDKError

load_dotenv(find_dotenv(), override=True)

from collections import defaultdict

def drop_running_headers_footers(elements, page_share=0.3, max_len=100):
    norm = lambda t: re.sub(r"\d+", "#", t.strip().lower())
    pages = {e["metadata"].get("page_number") for e in elements}
    seen = defaultdict(set)
    for e in elements:
        seen[norm(e["text"])].add(e["metadata"].get("page_number"))
    bad = {k for k, p in seen.items() if 6 < len(k) < max_len and len(p) >= page_share * len(pages)}
    return [e for e in elements if norm(e["text"]) not in bad]

def load_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    with UnstructuredClient(api_key_auth=os.getenv("UNSTRUCTURED_API_KEY")) as client:
        with open(pdf_path, "rb") as f:
            file_data = f.read()
            
        req = shared.PartitionParameters(
            files=shared.Files(
                content=file_data,
                file_name=os.path.basename(pdf_path),
            ),
            strategy=shared.Strategy.HI_RES,
            infer_table_structure=True,
            languages=["eng"],
        )

        req = operations.PartitionRequest(partition_parameters=req)
        
        try:
            print("1. Sending to Unstructured API...\n")
            response = client.general.partition(request=req)
            
            cleaned_elements = []
            seen_texts = set()

            cleaned_elements = drop_running_headers_footers(response.elements)

            print("2. Filtering footers, headers, duplicates, and junk...\n")


            for element in response.elements:
                
                if "text" not in element or not element["text"]:
                    continue

                if element.get("category") in ["Header", "Footer"]:
                    continue

                text = clean_extra_whitespace(element["text"]).strip()

                if re.search(r"^Page \d+ of \d+$", text, re.IGNORECASE):
                    continue

                if len(text) <= 2:
                    continue

                if text in seen_texts:
                    continue

                seen_texts.add(text)
                element["text"] = text
                cleaned_elements.append(element)

            print(f"Success! Cleaned {len(cleaned_elements)} elements.\n")
            return cleaned_elements
            
        except SDKError as e:
            print(f"SDK Error occurred: {e}")
            return None
