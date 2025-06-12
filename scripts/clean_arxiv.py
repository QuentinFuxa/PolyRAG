import pandas as pd
from sqlalchemy import create_engine, TEXT, TIMESTAMP, ARRAY
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.types import String, DateTime
import psycopg2

# Define your DataFrame
df = pd.read_pickle("all_papers.pkl")

# Convert date columns to datetime if not already
df["date_first_submission"] = pd.to_datetime(df["date_first_submission"])
df["date_last_submission"] = pd.to_datetime(df["date_last_submission"])
df["submission_dates"] = df["submission_dates"].apply(lambda x: pd.to_datetime(x))

# Define SQLAlchemy engine
user = "postgres"
host = "localhost"
port = "5432"
database = "demo_articles"
engine = create_engine(f"postgresql+psycopg2://{user}@{host}:{port}/{database}")

# Define type mapping for columns
dtype = {
    "link": TEXT(),
    "title": TEXT(),
    "date_first_submission": TIMESTAMP(),
    "date_last_submission": TIMESTAMP(),
    "submission_dates": PG_ARRAY(TIMESTAMP()),
    "authors": PG_ARRAY(TEXT()),
    "abstract": TEXT(),
    "primary_subject": TEXT(),
    "subjects": PG_ARRAY(TEXT()),
    "pdf_url": TEXT()
}

# Upload DataFrame with correct types
df.to_sql("arxiv_qbio", engine, if_exists="replace", index=False, dtype=dtype)