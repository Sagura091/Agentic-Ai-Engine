# Database Setup Scripts for Agentic AI System

This directory contains scripts to quickly restore your database after clearing Docker volumes or resetting the database.

## 🚀 Quick Start (Recommended)

For a fast, reliable setup that gets you running immediately:

```bash
python quick_setup_database.py
```

This creates the essential tables needed for basic functionality:
- ✅ User authentication (users, user_sessions)
- ✅ Agent management (agents)
- ✅ Conversations and messages
- ✅ Basic indexes for performance

## 🔧 Complete Setup (Advanced)

For a full setup with all features (may have compatibility issues):

```bash
python setup_database.py
```

This attempts to:
1. Execute `init-db.sql` for complete table structure
2. Run all database migrations
3. Set up advanced features

**Note**: This script may encounter issues with complex SQL parsing and Unicode characters on Windows.

## 📁 Files Overview

### Scripts
- **`quick_setup_database.py`** - ⭐ **Recommended** - Simple, reliable setup
- **`setup_database.py`** - Complete setup (may have issues)
- **`setup_database.sh`** - Shell script version (Linux/Mac)

### Database Files
- **`init-db.sql`** - Complete database initialization SQL
- **`db/migrations/run_all_migrations.py`** - Full migration system

## 🔄 When to Use These Scripts

Use these scripts when you need to restore your database after:

1. **Clearing Docker volumes**:
   ```bash
   docker-compose down -v  # This clears volumes
   docker-compose up -d    # Restart containers
   python quick_setup_database.py  # Restore database
   ```

2. **Database corruption or reset**
3. **Fresh development environment setup**
4. **Moving to a new machine**

## 📋 Step-by-Step Recovery Process

### 1. Ensure Database is Running
Make sure your PostgreSQL container is running:
```bash
docker-compose up -d postgres
```

### 2. Run Setup Script
```bash
# Quick setup (recommended)
python quick_setup_database.py

# OR complete setup (if you need all features)
python setup_database.py
```

### 3. Start the Application
```bash
# Backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (in another terminal)
cd frontend
npm run dev
```

### 4. Create Admin User
1. Visit: http://localhost:8000/register
2. Create your admin account
3. Manually set admin role if needed:
   ```sql
   UPDATE users SET role = 'admin' WHERE username = 'your_username';
   ```

## ⚙️ Configuration

### Database Connection
The scripts use these default connection settings:
- **Host**: localhost
- **Port**: 5432
- **Database**: agentic_ai
- **User**: agentic_user
- **Password**: agentic_password

You can override these with environment variables:
```bash
export DATABASE_URL="postgresql://user:password@host:port/database"
```

### Troubleshooting

#### Connection Issues
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check logs
docker-compose logs postgres
```

#### Permission Issues
```bash
# Make sure you're in the project root
ls -la  # Should see init-db.sql and db/ directory

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Unicode Issues (Windows)
If you see Unicode encoding errors, use the quick setup script instead:
```bash
python quick_setup_database.py
```

## 🎯 What Each Script Does

### Quick Setup (`quick_setup_database.py`)
- ✅ Creates essential tables only
- ✅ Reliable SQL execution
- ✅ Windows compatible
- ✅ Fast execution
- ❌ Limited to core features

### Complete Setup (`setup_database.py`)
- ✅ Full feature set
- ✅ Runs init-db.sql
- ✅ Executes all migrations
- ❌ May have parsing issues
- ❌ Unicode compatibility issues on Windows

## 🔍 Verification

After running either script, verify your setup:

```sql
-- Check tables exist
\dt

-- Check users table
SELECT * FROM users LIMIT 1;

-- Check agents table
SELECT * FROM agents LIMIT 1;
```

## 📞 Support

If you encounter issues:

1. **Try the quick setup first**: `python quick_setup_database.py`
2. **Check the logs** for specific error messages
3. **Verify database connection** manually
4. **Ensure you're in the project root directory**

## 🎉 Success!

Once setup is complete, you should be able to:
- ✅ Access the application at http://localhost:8000
- ✅ Register new users
- ✅ Create and manage agents
- ✅ Use the admin panel at http://localhost:5174/admin/enhanced-settings

Your database is now ready for use! 🚀
