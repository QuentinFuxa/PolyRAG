ALTER TABLE arxiv_qbio
ADD COLUMN arxiv_id TEXT;

UPDATE arxiv_qbio
SET arxiv_id = split_part(link, '/abs/', 2);