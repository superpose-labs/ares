# ARES React Dashboard - Quick Start Guide

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Node.js 18+ installed (`node --version`)
- [ ] Python 3.9+ installed (`python --version`)
- [ ] ARES data ingested (run `python main.py` from root)
- [ ] MongoDB running (`docker-compose -f mongo-docker-compose.yml up -d`)

## Quick Start (Development)

### 1. Start Backend (Terminal 1)

```bash
# From the project root
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start API server
python main.py
```

You should see:
```
✅ Database connections established
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Start Frontend (Terminal 2)

```bash
# From the project root
cd frontend

# Install Node dependencies
npm install

# Start development server
npm run dev
```

You should see:
```
  ▲ Next.js 14.x.x
  - Local:        http://localhost:3000
```

### 3. Open Browser

Navigate to `http://localhost:3000`

## Common Issues & Solutions

### Backend Won't Start

**Problem**: `ModuleNotFoundError: No module named 'ares'`

**Solution**: Install ARES package
```bash
cd ..  # Go to project root
pip install -e .
```

**Problem**: `Database not initialized`

**Solution**: Ensure data exists
```bash
cd ..  # Go to project root
python main.py  # Run ingestion first
```

### Frontend Won't Connect

**Problem**: API calls fail with network error

**Solution**: Check backend is running on port 8000
```bash
curl http://localhost:8000/
# Should return: {"message": "ARES Dashboard API", "status": "running"}
```

**Problem**: CORS errors in browser console

**Solution**: Verify backend CORS settings in `backend/main.py` include `http://localhost:3000`

### No Data Showing

**Problem**: Dashboard loads but shows 0 rollouts

**Solution**: Verify database has data
```bash
# From project root
python -c "from ares.databases.structured_database import StructuredDatabase; db = StructuredDatabase(); print(f'Rollouts: {len(db.get_all_as_dataframe())}')"
```

If output is 0, run ingestion:
```bash
python main.py
```

## Architecture Overview

```
┌─────────────────┐
│  Browser        │
│  localhost:3000 │
└────────┬────────┘
         │
         │ HTTP Requests
         ▼
┌─────────────────┐
│  FastAPI        │      ┌──────────────┐
│  localhost:8000 │◄────►│  SQLite      │
└─────────────────┘      │  MongoDB     │
                         │  FAISS       │
                         └──────────────┘
```

## File Structure (Key Files)

```
frontend/
├── src/app/page.tsx          # Main dashboard page
├── src/store/filterStore.ts  # State management
└── src/lib/api.ts            # API client

backend/
└── main.py                    # FastAPI server
```

## Next Steps

1. **Explore the dashboard**: Try filtering by success rate or robot type
2. **Select a rollout**: Use the dropdown in "Rollout Display" section
3. **Try embeddings**: Use lasso selection on UMAP plots
4. **Export data**: Download filtered results as CSV or JSON

## Need Help?

- Check logs in both terminal windows
- Open browser DevTools (F12) → Console tab
- Review the full README.md for detailed documentation

---

Enjoy exploring your robot data! 🤖✨
