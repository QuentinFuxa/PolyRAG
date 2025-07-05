

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re
from tqdm import tqdm
from sqlalchemy import create_engine

############################ 1 ############################


# def extract_date_from_doi(doi):
#     match = re.search(r'10\.1101/(\d{4})\.(\d{2})\.(\d{2})', doi)
#     if match:
#         return datetime.strptime(match.group(0).split("/")[1], "%Y.%m.%d")
#     return None

# def scrape_medrxiv(pages=92):
#     base_url = "https://www.medrxiv.org/search/%20jcode%3Amedrxiv%20limit_from%3A2025-01-01%20limit_to%3A2025-06-13%20numresults%3A75%20sort%3Apublication-date%20direction%3Adescending%20format_result%3Astandard?page="
#     data = []

#     for page in tqdm(range(1, pages + 1)):
#         url = base_url + str(page)
#         try:
#             res = requests.get(url, timeout=10)
#             res.raise_for_status()
#         except Exception as e:
#             print(f"Error on page {page}: {e}")
#             continue

#         soup = BeautifulSoup(res.text, "html.parser")
#         papers = soup.find_all("div", class_="highwire-cite")

#         for paper in papers:
#             a_tag = paper.select_one(".highwire-cite-title a")
#             doi_tag = paper.select_one(".highwire-cite-metadata-doi")

#             if not a_tag or not doi_tag:
#                 continue

#             title = a_tag.text.strip()
#             url = "https://www.medrxiv.org" + a_tag["href"]
#             doi = doi_tag.text.strip().replace("doi:", "").strip()
#             pub_date = extract_date_from_doi(doi)

#             data.append({
#                 "title": title,
#                 "url": url,
#                 "publication_date": pub_date,
#                 "doi": doi
#             })

#     return pd.DataFrame(data)

# df = scrape_medrxiv(pages=92)
# df.to_pickle("medrxiv_2025_01_01_to_06_13.pkl")
# print(f"Saved {len(df)} entries to pickle.")

############################ 2 ############################

# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# from tqdm import tqdm

# def extract_pdf_and_subject(url):
#     try:
#         res = requests.get(url, timeout=10)
#         res.raise_for_status()
#     except Exception as e:
#         print(f"Error fetching {url}: {e}")
#         return None, None

#     soup = BeautifulSoup(res.text, "html.parser")

#     # PDF link
#     pdf_tag = soup.select_one("a.article-dl-pdf-link")
#     pdf_url = f"https://www.medrxiv.org{pdf_tag['href']}" if pdf_tag else None

#     # Subject Area
#     subject_tag = soup.select_one("div.highwire-article-collections a")
#     subject = subject_tag.text.strip() if subject_tag else None

#     return pdf_url, subject

# # Example usage with an existing dataframe
# # Assume df has columns: title, publication_date, url
# df = pd.read_pickle("medrxiv_2025_01_01_to_06_13.pkl")

# pdf_urls = []
# subjects = []

# for url in tqdm(df["url"]):
#     pdf_url, subject = extract_pdf_and_subject(url)
#     pdf_urls.append(pdf_url)
#     subjects.append(subject)

# df["pdf_url"] = pdf_urls
# df["subject_area"] = subjects

# # Save enriched dataframe
# df.to_pickle("medrxiv_2025_enriched.pkl")


df = pd.read_pickle("medrxiv_2025_enriched.pkl")

user = "postgres"
host = "localhost"
port = "5432"
database = "demo_articles"

engine = create_engine(f"postgresql+psycopg2://{user}@{host}:{port}/{database}")

df.to_sql("medrxiv_2025", engine, if_exists="replace", index=False)
print(f"Saved {len(df)} entries to PostgreSQL database.")