# ARES Dashboard - React Web Application

A modern Next.js + Tailwind CSS web application that reproduces the functionality of the ARES Streamlit dashboard for robot data curation and evaluation.

## 🚀 Features

This React web application provides all the core functionality of the original Streamlit app:

### Data Management
- **Real-time Data Loading**: Connects to SQLite, MongoDB, and FAISS databases
- **State Management**: Efficient client-side state with Zustand
- **Caching**: React Query for optimized data fetching

### Filtering Capabilities
- **Structured Filters**:
  - Numeric range sliders (success rate, trajectory length, etc.)
  - Categorical multiselect (dataset, robot, environment, etc.)
  - NaN value handling
  - Real-time filter preview

- **Embedding-Based Filters**:
  - Interactive UMAP visualizations
  - Point, box, and lasso selection tools
  - Multiple embedding types (task, description, trajectory)
  - AND operation across filter types

### Analytics & Visualization
- **Distribution Analytics**: Auto-generated histograms and bar charts
- **Success Rate Analytics**: Grouped success metrics by category
- **Time Series Trends**: Temporal analysis of key metrics
- **Interactive Plotly Charts**: Fully interactive data exploration

### Display Components
- **Video Grid**: Preview up to 5 rollout examples
- **Hero Display**: Detailed view of individual rollouts with:
  - Full video playback
  - Complete metadata
  - Similar rollout recommendations
  - Annotation visualization

### Data Export
- Multiple formats: CSV, JSON, PDF, Parquet, HTML
- Client-side and server-side export options
- Customizable export contents

## 📁 Project Structure

```
ares/
├── frontend/                    # Next.js React application
│   ├── src/
│   │   ├── app/                # Next.js app directory
│   │   │   ├── globals.css     # Global styles
│   │   │   ├── layout.tsx      # Root layout
│   │   │   ├── page.tsx        # Main dashboard page
│   │   │   └── providers.tsx   # React Query provider
│   │   ├── components/         # React components
│   │   │   ├── layout/
│   │   │   │   └── DashboardHeader.tsx
│   │   │   ├── filters/
│   │   │   │   ├── StructuredFilters.tsx
│   │   │   │   └── EmbeddingFilters.tsx
│   │   │   ├── analytics/
│   │   │   │   ├── DataDistributions.tsx
│   │   │   │   ├── SuccessRateAnalytics.tsx
│   │   │   │   └── TimeSeriesAnalytics.tsx
│   │   │   ├── display/
│   │   │   │   ├── VideoGrid.tsx
│   │   │   │   └── HeroDisplay.tsx
│   │   │   └── export/
│   │   │       └── ExportOptions.tsx
│   │   ├── store/              # Zustand state management
│   │   │   └── filterStore.ts
│   │   ├── lib/                # Utilities
│   │   │   ├── api.ts          # API client
│   │   │   └── utils.ts        # Helper functions
│   │   └── types/              # TypeScript types
│   │       └── index.ts
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   └── next.config.mjs
│
└── backend/                     # FastAPI backend
    ├── main.py                  # API server
    └── requirements.txt
```

## 🛠️ Installation

### Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.9+
- MongoDB (for annotations)
- Existing ARES data directory with SQLite, MongoDB, and FAISS databases

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env.local file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Backend Setup

```bash
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make sure MongoDB is running
docker-compose -f ../mongo-docker-compose.yml up -d

# Start the API server
python main.py
```

The API will be available at `http://localhost:8000`

## 🚦 Usage

1. **Start the backend server** (must be running first):
   ```bash
   cd backend
   python main.py
   ```

2. **Start the frontend development server**:
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open your browser** to `http://localhost:3000`

4. **Explore the dashboard**:
   - Use structured filters to narrow down rollouts by numeric and categorical attributes
   - Use embedding filters to select clusters in UMAP space
   - View analytics and distributions
   - Select individual rollouts to see detailed information
   - Export filtered data in your preferred format

## 🎨 Technology Stack

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **React Query (@tanstack/react-query)**: Data fetching and caching
- **Zustand**: Lightweight state management
- **Plotly.js**: Interactive visualizations
- **Lucide React**: Icon library
- **date-fns**: Date formatting
- **papaparse**: CSV parsing
- **jsPDF**: PDF generation

### Backend
- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: SQL database ORM
- **pymongo**: MongoDB driver
- **faiss-cpu**: Vector similarity search
- **pandas**: Data manipulation
- **pyarrow**: Parquet file support

## 🔧 Configuration

### Environment Variables

#### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### Backend
The backend uses the existing ARES configuration from `src/ares/constants.py`. Make sure:
- `ARES_DATA_DIR` points to your data directory
- MongoDB is running on the default port (27017)
- Embedding cache files exist in `data/webapp_tmp/`

## 📊 Features Comparison

| Feature | Streamlit App | React App | Notes |
|---------|--------------|-----------|-------|
| Data Loading | ✅ | ✅ | Same database connections |
| Structured Filters | ✅ | ✅ | Numeric + categorical |
| Embedding Filters | ✅ | ✅ | UMAP with interactive selection |
| Distribution Charts | ✅ | ✅ | Auto-generated visualizations |
| Success Analytics | ✅ | ✅ | Grouped bar charts |
| Time Series | ✅ | ✅ | Line charts over time |
| Video Grid | ✅ | ✅ | Lazy loading supported |
| Hero Display | ✅ | ✅ | Full rollout details |
| Similar Rollouts | ✅ | ✅ | Multiple similarity metrics |
| Annotations | ✅ | ✅ | Frame-level data |
| Robot Arrays | ⚠️ | 🔄 | Planned for future |
| CSV Export | ✅ | ✅ | Client-side |
| JSON Export | ✅ | ✅ | Client-side |
| PDF Export | ✅ | ✅ | Basic version |
| Parquet Export | ✅ | ✅ | Server-side |
| HTML Export | ✅ | ✅ | Server-side |

## 🎯 Key Differences from Streamlit App

### Advantages
1. **Better Performance**:
   - Client-side state management reduces re-renders
   - React Query caching minimizes API calls
   - Progressive loading with suspense

2. **Modern UI/UX**:
   - Responsive design out of the box
   - Smoother animations and transitions
   - Better mobile support

3. **Scalability**:
   - Separate frontend/backend architecture
   - Can deploy to CDN
   - Horizontal scaling possible

4. **Developer Experience**:
   - TypeScript for type safety
   - Component reusability
   - Better debugging tools

### Considerations
1. **Setup Complexity**: Requires both frontend and backend servers
2. **Development Time**: More initial setup than Streamlit
3. **Deployment**: Needs separate hosting for frontend and backend

## 🚀 Production Deployment

### Frontend (Vercel/Netlify)

```bash
cd frontend
npm run build
npm run start
```

Or deploy to Vercel:
```bash
npx vercel
```

### Backend (Docker)

Create `backend/Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t ares-backend .
docker run -p 8000:8000 ares-backend
```

## 🐛 Troubleshooting

### Frontend Issues

**Error: Cannot connect to API**
- Ensure backend is running on `http://localhost:8000`
- Check `.env.local` file has correct `NEXT_PUBLIC_API_URL`

**Plotly not rendering**
- This is normal during SSR - charts load after hydration
- Check browser console for errors

### Backend Issues

**Error: Database not found**
- Ensure `ARES_DATA_DIR` environment variable is set
- Run ingestion pipeline first: `python main.py`

**Error: MongoDB connection failed**
- Start MongoDB: `docker-compose -f mongo-docker-compose.yml up -d`
- Check MongoDB is running: `docker ps`

**Error: UMAP cache not found**
- Run the Streamlit app once to generate caches
- Or implement UMAP generation in backend

## 📝 Development

### Adding New Features

1. **New Filter Type**:
   - Add to `filterStore.ts`
   - Create component in `components/filters/`
   - Update API endpoint in `backend/main.py`

2. **New Visualization**:
   - Create component in `components/analytics/`
   - Add to main page in `app/page.tsx`

3. **New Export Format**:
   - Add handler in `ExportOptions.tsx`
   - Add API endpoint in `backend/main.py`

### Code Style

```bash
# Frontend
npm run lint
npm run format

# Backend
black backend/
flake8 backend/
```

## 🤝 Contributing

This web app is part of the ARES project. Follow the main project's contribution guidelines.

## 📄 License

Same license as the main ARES project (see LICENSE file in root directory).

## 🙏 Acknowledgments

This React web application is a reproduction of the original Streamlit dashboard, maintaining feature parity while leveraging modern web technologies for improved performance and scalability.
