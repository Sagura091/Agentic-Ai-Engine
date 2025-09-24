#!/usr/bin/env python3
"""
Fix the admin_setting_history table by adding the missing applied_at column.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.models.database.base import get_database_session

async def fix_admin_history_table():
    """Add the missing applied_at column to admin_setting_history table."""
    
    try:
        print("🔧 Fixing admin_setting_history table...")
        
        async for session in get_database_session():
            # Add the missing applied_at column
            await session.execute(text("""
                ALTER TABLE admin_setting_history 
                ADD COLUMN IF NOT EXISTS applied_at TIMESTAMP WITH TIME ZONE;
            """))
            
            await session.commit()
            print("✅ Added applied_at column to admin_setting_history table")
            break  # Exit after first session
            
        print("🎉 Table fix completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing table: {str(e)}")
        return False

async def main():
    """Main function."""
    print("🔧 Admin Settings History Table Fix")
    print("=" * 50)
    
    success = await fix_admin_history_table()
    if not success:
        return 1
    
    print("\n🎉 SUCCESS! Table is now fixed!")
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
