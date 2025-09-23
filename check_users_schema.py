import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from app.models.database.base import get_engine
from sqlalchemy import text

async def check_users_schema():
    engine = get_engine()
    async with engine.begin() as conn:
        # Check users table schema
        result = await conn.execute(text("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'users' AND table_schema = 'public'
            ORDER BY ordinal_position
        """))
        columns = result.fetchall()
        print('Users table schema:')
        for col in columns:
            nullable = "NULL" if col.is_nullable == "YES" else "NOT NULL"
            default = f" DEFAULT {col.column_default}" if col.column_default else ""
            print(f'  - {col.column_name}: {col.data_type} {nullable}{default}')
        
        # Check if users table is empty
        result = await conn.execute(text('SELECT COUNT(*) FROM users'))
        count = result.scalar()
        print(f'\nUsers in table: {count}')

if __name__ == "__main__":
    asyncio.run(check_users_schema())
