#!/usr/bin/env python3
"""
Verify and fix database schema issues.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.models.database.base import get_database_session

async def verify_and_fix_database():
    """Verify and fix database schema issues."""
    
    try:
        print("🔍 Verifying database schema...")
        
        async for session in get_database_session():
            # Check if applied_at column exists
            result = await session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'admin_setting_history' 
                AND column_name = 'applied_at';
            """))
            applied_at_exists = result.fetchone() is not None
            
            if not applied_at_exists:
                print("❌ applied_at column missing, adding it...")
                await session.execute(text("""
                    ALTER TABLE admin_setting_history 
                    ADD COLUMN applied_at TIMESTAMP WITH TIME ZONE;
                """))
                print("✅ Added applied_at column")
            else:
                print("✅ applied_at column exists")
            
            # Check table structure
            result = await session.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'admin_setting_history'
                ORDER BY ordinal_position;
            """))
            columns = result.fetchall()
            
            print("\n📋 admin_setting_history table structure:")
            for col in columns:
                print(f"  - {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})")
            
            # Check if we have any settings
            result = await session.execute(text("SELECT COUNT(*) FROM admin_settings;"))
            settings_count = result.scalar()
            print(f"\n📊 Found {settings_count} settings in admin_settings table")
            
            await session.commit()
            break  # Exit after first session
            
        print("\n🎉 Database verification completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying database: {str(e)}")
        return False

async def main():
    """Main function."""
    print("🔧 Database Schema Verification and Fix")
    print("=" * 50)
    
    success = await verify_and_fix_database()
    if not success:
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        sys.exit(1)
