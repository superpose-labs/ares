from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any
import sys
import os
from pathlib import Path

# Add parent directory to path to import ares modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ares.databases import structured_database as struct_db
from ares.constants import ARES_DATA_DIR
import pandas as pd
import json
import io

app = FastAPI(title="ARES Dashboard API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database engine (will be initialized on startup)
db_engine = None

# Try to import optional databases
try:
    from ares.databases.annotation_database import AnnotationDatabase
    annotation_db = None
except ImportError:
    AnnotationDatabase = None
    annotation_db = None

try:
    from ares.databases.embedding_database import EmbeddingDatabase
    embedding_db = None
except ImportError:
    EmbeddingDatabase = None
    embedding_db = None


@app.on_event("startup")
async def startup_event():
    """Initialize database connections on startup"""
    global db_engine, annotation_db, embedding_db

    try:
        db_engine = struct_db.setup_database(struct_db.RolloutSQLModel)
        print("✅ StructuredDatabase connected")
    except Exception as e:
        print(f"❌ Error: Could not connect to StructuredDatabase: {e}")
        raise

    if AnnotationDatabase:
        try:
            annotation_db = AnnotationDatabase()
            print("✅ AnnotationDatabase connected")
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to AnnotationDatabase (MongoDB): {e}")
            print("   Annotations will not be available")

    if EmbeddingDatabase:
        try:
            embedding_db = EmbeddingDatabase()
            print("✅ EmbeddingDatabase connected")
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to EmbeddingDatabase: {e}")
            print("   Embedding filters will not be available")


@app.get("/")
async def root():
    return {"message": "ARES Dashboard API", "status": "running"}


@app.get("/api/rollouts")
async def get_rollouts():
    """Fetch all rollouts from the structured database"""
    if not db_engine:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        df = struct_db.get_rollouts_as_df(db_engine)
        # Convert to records and handle NaN values
        records = df.where(pd.notna(df), None).to_dict('records')
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/embeddings/{embedding_type}")
async def get_embeddings(embedding_type: str):
    """Get UMAP projections and clusters for embeddings"""
    try:
        # Load pre-computed UMAP from cache
        cache_path = Path(ARES_DATA_DIR) / "webapp_tmp" / f"{embedding_type}_umap.npz"

        if cache_path.exists():
            import numpy as np
            data = np.load(cache_path)
            return {
                "umap_x": data['umap_x'].tolist(),
                "umap_y": data['umap_y'].tolist(),
                "ids": data['ids'].tolist(),
                "clusters": data['clusters'].tolist(),
                "labels": data.get('labels', ['']*len(data['ids'])).tolist(),
            }
        else:
            return {
                "umap_x": [],
                "umap_y": [],
                "ids": [],
                "clusters": [],
                "labels": [],
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/annotations/{rollout_id}")
async def get_annotations(rollout_id: str):
    """Fetch annotations for a specific rollout"""
    if not annotation_db:
        return {"error": "Annotation database not available"}

    try:
        annotations = annotation_db.get_annotations(rollout_id)
        return annotations or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SimilarityRequest(BaseModel):
    rollout_id: str
    similarity_type: str
    n: int = 10


@app.post("/api/similarity")
async def get_similar_rollouts(request: SimilarityRequest):
    """Find similar rollouts based on embeddings"""
    if not db_engine:
        raise HTTPException(status_code=500, detail="Database not initialized")

    try:
        # Get the rollout
        df = struct_db.get_rollouts_as_df(db_engine)
        rollout = df[df['id'] == request.rollout_id]

        if rollout.empty:
            raise HTTPException(status_code=404, detail="Rollout not found")

        # For now, return similar rollouts based on simple sampling
        # In production, this would use FAISS similarity search
        similar = df[df['id'] != request.rollout_id].sample(min(request.n, len(df)-1))
        records = similar.where(pd.notna(similar), None).to_dict('records')
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExportRequest(BaseModel):
    data: List[dict]
    format: str


@app.post("/api/export")
async def export_data(request: ExportRequest):
    """Export data in various formats"""
    try:
        df = pd.DataFrame(request.data)

        if request.format == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=export.csv"}
            )

        elif request.format == "parquet":
            output = io.BytesIO()
            df.to_parquet(output, index=False)
            output.seek(0)
            return StreamingResponse(
                output,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename=export.parquet"}
            )

        elif request.format == "json":
            output = df.to_json(orient='records', indent=2)
            return StreamingResponse(
                io.BytesIO(output.encode()),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=export.json"}
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos/{video_path:path}")
async def get_video(video_path: str):
    """Serve video files"""
    video_full_path = Path(ARES_DATA_DIR) / "videos" / video_path

    if not video_full_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        video_full_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
