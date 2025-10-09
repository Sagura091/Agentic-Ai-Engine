"""
005_add_document_tables.py
Add revolutionary document storage tables for RAG system

Migration Order: 005 (Fifth - Document Storage & RAG)
Dependencies: 004_create_enhanced_tables.py
Next: None (Final migration)

Original Revision ID: 001_add_document_tables
Create Date: 2025-09-16 14:45:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_add_document_tables'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create revolutionary document storage tables."""
    
    # Create rag schema if it doesn't exist
    op.execute('CREATE SCHEMA IF NOT EXISTS rag')
    
    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('knowledge_base_id', sa.String(255), nullable=False, index=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('original_filename', sa.String(255), nullable=False),
        sa.Column('content_type', sa.String(100), nullable=False),
        sa.Column('file_size', sa.Integer, nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False, index=True),
        sa.Column('encrypted_content', sa.LargeBinary, nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('processing_error', sa.Text, nullable=True),
        sa.Column('chunk_count', sa.Integer, nullable=False, default=0),
        sa.Column('embedding_model', sa.String(100), nullable=True),
        sa.Column('embedding_dimensions', sa.Integer, nullable=True),
        sa.Column('document_type', sa.String(50), nullable=False, default='text'),
        sa.Column('language', sa.String(10), nullable=False, default='en'),
        sa.Column('doc_metadata', postgresql.JSONB, nullable=False, default={}),
        sa.Column('is_public', sa.Boolean, nullable=False, default=False),
        sa.Column('uploaded_by', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('processed_at', sa.DateTime, nullable=True),
        schema='rag'
    )
    
    # Create document_chunks table
    op.create_table(
        'document_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), 
                 sa.ForeignKey('rag.documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_index', sa.Integer, nullable=False),
        sa.Column('chunk_text', sa.Text, nullable=False),
        sa.Column('start_char', sa.Integer, nullable=False),
        sa.Column('end_char', sa.Integer, nullable=False),
        sa.Column('chromadb_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('collection_name', sa.String(255), nullable=False),
        sa.Column('embedding_model', sa.String(100), nullable=True),
        sa.Column('embedding_dimensions', sa.Integer, nullable=True),
        sa.Column('chunk_metadata', postgresql.JSONB, nullable=False, default={}),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        schema='rag'
    )
    
    # Create indexes for performance
    op.create_index('idx_documents_kb_id', 'documents', ['knowledge_base_id'], schema='rag')
    op.create_index('idx_documents_status', 'documents', ['status'], schema='rag')
    op.create_index('idx_documents_created_at', 'documents', ['created_at'], schema='rag')
    op.create_index('idx_documents_content_hash', 'documents', ['content_hash'], schema='rag')
    op.create_index('idx_document_chunks_doc_id', 'document_chunks', ['document_id'], schema='rag')
    op.create_index('idx_document_chunks_chromadb_id', 'document_chunks', ['chromadb_id'], schema='rag')
    
    # Create GIN index for JSONB metadata search
    op.create_index('idx_documents_metadata_gin', 'documents', ['doc_metadata'],
                   postgresql_using='gin', schema='rag')
    op.create_index('idx_document_chunks_metadata_gin', 'document_chunks', ['chunk_metadata'],
                   postgresql_using='gin', schema='rag')


def downgrade():
    """Drop revolutionary document storage tables."""
    
    # Drop indexes
    op.drop_index('idx_document_chunks_metadata_gin', schema='rag')
    op.drop_index('idx_documents_metadata_gin', schema='rag')
    op.drop_index('idx_document_chunks_chromadb_id', schema='rag')
    op.drop_index('idx_document_chunks_doc_id', schema='rag')
    op.drop_index('idx_documents_content_hash', schema='rag')
    op.drop_index('idx_documents_created_at', schema='rag')
    op.drop_index('idx_documents_status', schema='rag')
    op.drop_index('idx_documents_kb_id', schema='rag')
    
    # Drop tables
    op.drop_table('document_chunks', schema='rag')
    op.drop_table('documents', schema='rag')
    
    # Drop schema if empty
    op.execute('DROP SCHEMA IF EXISTS rag CASCADE')
