#!/usr/bin/env python3
"""
Fix database tables by dropping and recreating them.
"""

import asyncio
from app.models.database.base import get_engine, Base
from sqlalchemy import text

async def fix_database():
    try:
        engine = get_engine()
        
        # Check existing tables
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
            tables = [row[0] for row in result.fetchall()]
            print(f'Existing tables: {tables}')
            
            if 'users' in tables:
                print('Dropping existing users table...')
                await conn.execute(text('DROP TABLE IF EXISTS users CASCADE;'))
                print('✅ Users table dropped')
            
        print('Recreating all tables...')
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print('✅ All tables recreated successfully!')
        
        # Verify the users table has the correct columns
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'users';"))
            columns = [row[0] for row in result.fetchall()]
            print(f'Users table columns: {columns}')
            
            if 'user_group' in columns:
                print('✅ user_group column exists!')
            else:
                print('❌ user_group column missing!')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(fix_database())
