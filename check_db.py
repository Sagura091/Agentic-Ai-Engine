import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from app.models.database.base import get_engine
from sqlalchemy import text

async def check_database():
    engine = get_engine()
    async with engine.begin() as conn:
        # Check what tables exist
        result = await conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """))
        tables = result.fetchall()
        print(f'Tables in database: {[t.table_name for t in tables]}')
        
        # Check if users table exists and has data
        if any(t.table_name == 'users' for t in tables):
            result = await conn.execute(text('SELECT COUNT(*) FROM users'))
            count = result.scalar()
            print(f'Users in database: {count}')
            
            if count > 0:
                result = await conn.execute(text('SELECT username, email, user_group FROM users'))
                users = result.fetchall()
                for user in users:
                    print(f'  - {user.username} ({user.email}) - {user.user_group}')
        else:
            print('No users table found')

if __name__ == "__main__":
    asyncio.run(check_database())
