#!/usr/bin/env python3
"""
Quick script to check admin user status and create one if needed.
"""
import asyncio
import sys
import os
sys.path.append('.')

from app.core.database import get_database_session
from sqlalchemy import text

async def check_and_create_admin():
    try:
        async with get_database_session() as session:
            # Check if users table exists
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'users'
                );
            """))
            table_exists = result.scalar()
            print(f'Users table exists: {table_exists}')
            
            if not table_exists:
                print('Creating users table...')
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS users (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        username VARCHAR(255) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        name VARCHAR(255),
                        hashed_password VARCHAR(255) NOT NULL,
                        password_salt VARCHAR(255),
                        is_active BOOLEAN DEFAULT true,
                        user_group VARCHAR(50) DEFAULT 'user' CHECK (user_group IN ('user', 'moderator', 'admin')),
                        failed_login_attempts INTEGER DEFAULT 0,
                        locked_until TIMESTAMP WITH TIME ZONE,
                        last_login TIMESTAMP WITH TIME ZONE,
                        login_count INTEGER DEFAULT 0,
                        api_keys JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """))
                await session.commit()
                print('Users table created')
            
            # Check for admin user
            result = await session.execute(text("""
                SELECT username, email, user_group, is_active 
                FROM users 
                WHERE username = 'admin' OR user_group = 'admin'
                LIMIT 5;
            """))
            users = result.fetchall()
            print(f'Admin users found: {len(users)}')
            for user in users:
                print(f'  - {user.username} ({user.email}) - {user.user_group} - Active: {user.is_active}')
            
            if len(users) == 0:
                print('No admin user found. Creating default admin user...')
                # Create admin user with hashed password
                import hashlib
                import secrets
                
                password = 'admin123'
                salt = secrets.token_hex(16)
                hashed_password = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
                
                await session.execute(text("""
                    INSERT INTO users (username, email, name, hashed_password, password_salt, user_group, is_active)
                    VALUES ('admin', 'admin@localhost', 'System Administrator', :hashed_password, :salt, 'admin', true)
                """), {
                    'hashed_password': hashed_password,
                    'salt': salt
                })
                await session.commit()
                print('Admin user created successfully!')
                print('Username: admin')
                print('Email: admin@localhost')
                print('Password: admin123')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_and_create_admin())
