'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Eye, Video, ChevronDown, ChevronUp } from 'lucide-react';
import { Rollout } from '@/types';
import { useFilterStore } from '@/store/filterStore';
import { fetchAnnotations, fetchSimilarRollouts } from '@/lib/api';

interface HeroDisplayProps {
  data: Rollout[];
}

export function HeroDisplay({ data }: HeroDisplayProps) {
  const { selectedRow, setSelectedRow } = useFilterStore();
  const [showDetails, setShowDetails] = useState(false);
  const [similarityType, setSimilarityType] = useState('task_embedding');

  const handleRowSelection = (index: number) => {
    if (data[index]) {
      setSelectedRow(data[index]);
    }
  };

  const { data: annotations } = useQuery({
    queryKey: ['annotations', selectedRow?.id],
    queryFn: () => (selectedRow ? fetchAnnotations(selectedRow.id) : null),
    enabled: !!selectedRow,
  });

  const { data: similarRollouts } = useQuery({
    queryKey: ['similar', selectedRow?.id, similarityType],
    queryFn: () =>
      selectedRow ? fetchSimilarRollouts(selectedRow.id, similarityType, 10) : null,
    enabled: !!selectedRow,
  });

  if (data.length === 0) {
    return null;
  }

  return (
    <section className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center space-x-2 mb-6">
        <Eye className="h-5 w-5 text-gray-500" />
        <h2 className="text-2xl font-bold text-gray-900">Rollout Display</h2>
      </div>

      {/* Row Selection */}
      <div className="mb-6 space-y-2">
        <label className="block text-sm font-medium text-gray-700">Select Rollout</label>
        <div className="flex space-x-2">
          <select
            onChange={(e) => handleRowSelection(parseInt(e.target.value))}
            className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
          >
            <option value="">Choose a rollout...</option>
            {data.slice(0, 100).map((row, idx) => (
              <option key={row.id} value={idx}>
                {row.task_language_instruction?.substring(0, 80) || `Rollout ${idx + 1}`}
              </option>
            ))}
          </select>
          <button
            onClick={() => handleRowSelection(Math.floor(Math.random() * data.length))}
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
          >
            Random
          </button>
        </div>
      </div>

      {selectedRow ? (
        <div className="space-y-6">
          {/* Selected Row Info */}
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600 mb-2">
              <span className="font-medium">Rollout ID:</span> {selectedRow.id}
            </p>
          </div>

          {/* Video and Metadata */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Video */}
            <div>
              {selectedRow.video_path ? (
                <video
                  src={`/api/videos/${selectedRow.video_path}`}
                  controls
                  className="w-full rounded-lg shadow-md"
                />
              ) : (
                <div className="w-full aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                  <Video className="h-16 w-16 text-gray-400" />
                </div>
              )}
            </div>

            {/* Right: Metadata */}
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Task</h3>
                <p className="text-gray-700">{selectedRow.task_language_instruction}</p>
              </div>

              {selectedRow.success_estimate !== null &&
                selectedRow.success_estimate !== undefined && (
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">Success Rate</h3>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-4 overflow-hidden">
                        <div
                          className="bg-green-500 h-full"
                          style={{ width: `${selectedRow.success_estimate * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-700">
                        {(selectedRow.success_estimate * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                )}

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-700">Dataset:</span>
                  <p className="text-gray-600">{selectedRow.dataset_name}</p>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Robot:</span>
                  <p className="text-gray-600">{selectedRow.robot_embodiment}</p>
                </div>
                {selectedRow.robot_color_estimate && (
                  <div>
                    <span className="font-medium text-gray-700">Color:</span>
                    <p className="text-gray-600">{selectedRow.robot_color_estimate}</p>
                  </div>
                )}
                {selectedRow.environment_surface_estimate && (
                  <div>
                    <span className="font-medium text-gray-700">Surface:</span>
                    <p className="text-gray-600">{selectedRow.environment_surface_estimate}</p>
                  </div>
                )}
                {selectedRow.environment_lighting_estimate && (
                  <div>
                    <span className="font-medium text-gray-700">Lighting:</span>
                    <p className="text-gray-600">{selectedRow.environment_lighting_estimate}</p>
                  </div>
                )}
                {selectedRow.trajectory_length && (
                  <div>
                    <span className="font-medium text-gray-700">Trajectory Length:</span>
                    <p className="text-gray-600">{selectedRow.trajectory_length}</p>
                  </div>
                )}
              </div>

              {/* Details Toggle */}
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="flex items-center space-x-2 text-primary-600 hover:text-primary-700 font-medium text-sm"
              >
                {showDetails ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                <span>{showDetails ? 'Hide' : 'Show'} Full Details</span>
              </button>

              {showDetails && (
                <div className="bg-gray-50 rounded-lg p-4 max-h-96 overflow-y-auto">
                  <pre className="text-xs text-gray-700 whitespace-pre-wrap">
                    {JSON.stringify(selectedRow, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>

          {/* Similar Rollouts */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Similar Rollouts</h3>
              <select
                value={similarityType}
                onChange={(e) => setSimilarityType(e.target.value)}
                className="text-sm rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              >
                <option value="task_embedding">Task Embedding</option>
                <option value="description_embedding">Description Embedding</option>
                <option value="trajectory_embedding">Trajectory Embedding</option>
                <option value="text_similarity">Text Similarity</option>
              </select>
            </div>

            {similarRollouts && similarRollouts.length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {similarRollouts.map((similar: Rollout) => (
                  <button
                    key={similar.id}
                    onClick={() => setSelectedRow(similar)}
                    className="border border-gray-200 rounded-lg p-3 hover:shadow-md transition-shadow text-left"
                  >
                    <p className="text-xs text-gray-700 line-clamp-3 mb-2">
                      {similar.task_language_instruction}
                    </p>
                    <p className="text-xs text-gray-500">{similar.dataset_name}</p>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-sm">Loading similar rollouts...</p>
            )}
          </div>

          {/* Annotations */}
          {annotations && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Annotations</h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <pre className="text-xs text-gray-700 whitespace-pre-wrap">
                  {JSON.stringify(annotations, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-12">
          <Eye className="h-12 w-12 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-500">Select a rollout to view details</p>
        </div>
      )}
    </section>
  );
}
