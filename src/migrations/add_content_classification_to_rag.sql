-- Migration to add content classification columns to rag_document_blocks
-- This enables integration of demand extraction functionality into the RAG system

-- Add columns for content classification
ALTER TABLE app_data.rag_document_blocks 
ADD COLUMN IF NOT EXISTS content_type TEXT DEFAULT 'regular',
ADD COLUMN IF NOT EXISTS section_type TEXT,
ADD COLUMN IF NOT EXISTS demand_priority INTEGER;

-- Add indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_rag_blocks_content_type 
ON app_data.rag_document_blocks(content_type);

CREATE INDEX IF NOT EXISTS idx_rag_blocks_section_type 
ON app_data.rag_document_blocks(section_type);

CREATE INDEX IF NOT EXISTS idx_rag_blocks_demand_priority 
ON app_data.rag_document_blocks(demand_priority);

-- Create index for combined queries
CREATE INDEX IF NOT EXISTS idx_rag_blocks_content_classification
ON app_data.rag_document_blocks(content_type, section_type, demand_priority);

-- Add comments for documentation
COMMENT ON COLUMN app_data.rag_document_blocks.content_type IS 
'Type of content: section_header, demand, regular';

COMMENT ON COLUMN app_data.rag_document_blocks.section_type IS 
'Section type if applicable: synthesis, demands, demandes_prioritaires, autres_demandes, information, observations, introduction, conclusion';

COMMENT ON COLUMN app_data.rag_document_blocks.demand_priority IS 
'Priority of demand if content_type is demand: 1 for priority demands, 2 for complementary demands';
