import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import re
from datetime import datetime
from sqlalchemy import create_engine


BASE_URL = "https://arxiv.org"
CATEGORY_URL = "https://arxiv.org/list/q-bio/2025?skip={skip}&show=50"

def extract_submission_dates(soup):
    history_div = soup.find("div", class_="submission-history")
    if not history_div:
        return []
    date_strings = re.findall(r'\b\w{3},\s\d{1,2}\s\w{3}\s\d{4}\s[\d:]+', history_div.get_text())
    dates = []
    for ds in date_strings:
        try:
            d = datetime.strptime(ds, '%a, %d %b %Y %H:%M:%S')
            dates.append(d)
        except Exception:
            continue
    return dates

def extract_pdf_url(soup, paper_id):
    pdf_a = soup.find("a", href=f"/pdf/{paper_id}")
    if pdf_a:
        return BASE_URL + pdf_a["href"]
    return ""

def extract_paper_meta(meta_div):
    title = meta_div.find("div", class_="list-title").get_text(strip=True).replace('Title:', '').strip()
    authors_div = meta_div.find("div", class_="list-authors")
    authors = [a.get_text(strip=True) for a in authors_div.find_all("a")]
    subj_div = meta_div.find("div", class_="list-subjects")
    subjects_raw = subj_div.get_text(strip=True).replace('Subjects:', '').strip()
    primary_subject = subj_div.find("span", class_="primary-subject")
    primary_subject = primary_subject.get_text(strip=True) if primary_subject else None
    subjects = [primary_subject] if primary_subject else []
    others = [s.strip() for s in re.split(r';', subjects_raw)]
    if primary_subject:
        subjects = [s for s in others if s]  # All subjects (first is primary)
    else:
        subjects = [s for s in others if s]
    return title, authors, primary_subject, subjects

all_papers = []

for skip in tqdm(range(0, 2001, 50)):
    url = CATEGORY_URL.format(skip=skip)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    # Get each paper meta & id
    for dt, meta_div in zip(soup.find_all("dt"), soup.find_all("div", class_="meta")):
        # Paper ID from the "abs" link
        abs_link = dt.find("a", title="Abstract")
        if not abs_link:
            continue
        paper_id = abs_link["href"].split("/")[-1]
        abs_url = f"{BASE_URL}/abs/{paper_id}"
        # Meta from list page
        title, authors, primary_subject, subjects = extract_paper_meta(meta_div)

        # Now parse abstract page for the rest
        abs_resp = requests.get(abs_url)
        abs_soup = BeautifulSoup(abs_resp.text, "html.parser")
        # Abstract
        abstract = abs_soup.find("blockquote", class_="abstract").get_text(strip=True).replace('Abstract:', '').strip()
        # Submission dates
        submission_dates = extract_submission_dates(abs_soup)
        first_date = submission_dates[0] if submission_dates else None
        last_date = submission_dates[-1] if submission_dates else None
        # PDF
        pdf_url = extract_pdf_url(abs_soup, paper_id)
        # Comments
        comments = ""
        comm_div = abs_soup.find("td", class_="tablecell comments")
        if comm_div:
            comments = comm_div.get_text(strip=True)

        all_papers.append({
            "link": abs_url,
            "title": title,
            "date_first_submission": first_date,
            "date_last_submission": last_date,
            "submission_dates": submission_dates,
            "authors": authors,
            "abstract": abstract,
            "primary_subject": primary_subject,
            "subjects": subjects,
            "pdf_url": pdf_url,
            "comments": comments,
        })
        time.sleep(0.5)

df = pd.DataFrame(all_papers)

user = "postgres"
host = "localhost"
port = "5432"
database = "demo_articles"

engine = create_engine(f"postgresql+psycopg2://{user}@{host}:{port}/{database}")

df.to_sql("arxiv_qbio", engine, if_exists="replace", index=False)