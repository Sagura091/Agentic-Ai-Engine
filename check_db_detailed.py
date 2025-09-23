import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from app.models.database.base import get_engine
from sqlalchemy import text

async def check_database_detailed():
    engine = get_engine()
    async with engine.begin() as conn:
        # Check what schemas exist
        result = await conn.execute(text("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            ORDER BY schema_name
        """))
        schemas = result.fetchall()
        print(f'Schemas: {[s.schema_name for s in schemas]}')
        
        # Check what tables exist in public schema
        result = await conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """))
        tables = result.fetchall()
        print(f'Tables in public schema: {[t.table_name for t in tables]}')
        
        # Check what tables exist in rag schema if it exists
        if any(s.schema_name == 'rag' for s in schemas):
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'rag' 
                ORDER BY table_name
            """))
            rag_tables = result.fetchall()
            print(f'Tables in rag schema: {[t.table_name for t in rag_tables]}')

if __name__ == "__main__":
    asyncio.run(check_database_detailed())
