'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Scatter } from 'lucide-react';
import { useFilterStore } from '@/store/filterStore';
import { fetchEmbeddings } from '@/lib/api';
import { PlotlyChart } from '@/components/charts/PlotlyChart';

const EMBEDDING_TYPES = [
  { key: 'task', label: 'Task Instructions' },
  { key: 'description', label: 'Descriptions' },
  { key: 'trajectory', label: 'Trajectories' },
];

export function EmbeddingFilters() {
  const [activeType, setActiveType] = useState('task');
  const { addEmbeddingSelection } = useFilterStore();

  const { data: embeddingData, isLoading } = useQuery({
    queryKey: ['embeddings', activeType],
    queryFn: () => fetchEmbeddings(activeType),
  });

  const handleSelection = (event: any) => {
    if (event.points) {
      const selectedIds = event.points.map((p: any) => p.customdata);
      addEmbeddingSelection({
        type: activeType as any,
        selectedIds,
      });
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-2">
        <Scatter className="h-5 w-5 text-gray-500" />
        <h3 className="text-lg font-semibold text-gray-900">Embedding-Based Filters</h3>
      </div>

      {/* Embedding Type Tabs */}
      <div className="flex space-x-2 border-b border-gray-200">
        {EMBEDDING_TYPES.map((type) => (
          <button
            key={type.key}
            onClick={() => setActiveType(type.key)}
            className={`px-4 py-2 font-medium text-sm transition-colors ${
              activeType === type.key
                ? 'border-b-2 border-primary-600 text-primary-600'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            {type.label}
          </button>
        ))}
      </div>

      {/* UMAP Visualization */}
      <div className="bg-gray-50 rounded-lg p-4">
        {isLoading ? (
          <div className="h-96 flex items-center justify-center">
            <p className="text-gray-500">Loading embeddings...</p>
          </div>
        ) : embeddingData ? (
          <PlotlyChart
            data={[
              {
                x: embeddingData.umap_x,
                y: embeddingData.umap_y,
                mode: 'markers',
                type: 'scatter',
                customdata: embeddingData.ids,
                marker: {
                  size: 5,
                  color: embeddingData.clusters,
                  colorscale: 'Viridis',
                  showscale: true,
                },
                text: embeddingData.labels,
                hovertemplate: '%{text}<extra></extra>',
              },
            ]}
            layout={{
              title: `UMAP Projection - ${EMBEDDING_TYPES.find((t) => t.key === activeType)?.label}`,
              xaxis: { title: 'UMAP 1' },
              yaxis: { title: 'UMAP 2' },
              height: 500,
              dragmode: 'lasso',
              hovermode: 'closest',
            }}
            config={{
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToAdd: ['select2d', 'lasso2d'],
            }}
            onSelected={handleSelection}
            className="w-full"
          />
        ) : (
          <div className="h-96 flex items-center justify-center">
            <p className="text-gray-500">No embedding data available</p>
          </div>
        )}
      </div>

      <div className="text-sm text-gray-600 space-y-1">
        <p className="font-medium">Selection Instructions:</p>
        <ul className="list-disc list-inside space-y-1 ml-2">
          <li>Click "Box Select" or "Lasso Select" in the toolbar above</li>
          <li>Draw around points to select them</li>
          <li>Double-click to clear selection</li>
          <li>Selections from different tabs are combined with AND operation</li>
        </ul>
      </div>
    </div>
  );
}
