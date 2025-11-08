'use client';

import { useMemo } from 'react';
import { Video } from 'lucide-react';
import { Rollout } from '@/types';

interface VideoGridProps {
  data: Rollout[];
}

export function VideoGrid({ data }: VideoGridProps) {
  const displayVideos = useMemo(() => {
    // Get one video per dataset, up to 5 videos
    const byDataset = data.reduce((acc: Record<string, Rollout>, row) => {
      if (!acc[row.dataset_name]) {
        acc[row.dataset_name] = row;
      }
      return acc;
    }, {});

    const videos = Object.values(byDataset).slice(0, 5);

    // Fill remaining slots if needed
    if (videos.length < 5) {
      const extra = data
        .filter((row) => !videos.find((v) => v.id === row.id))
        .slice(0, 5 - videos.length);
      videos.push(...extra);
    }

    return videos;
  }, [data]);

  if (displayVideos.length === 0) {
    return null;
  }

  return (
    <section className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center space-x-2 mb-6">
        <Video className="h-5 w-5 text-gray-500" />
        <h2 className="text-2xl font-bold text-gray-900">Rollout Examples</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {displayVideos.map((rollout) => (
          <VideoCard key={rollout.id} rollout={rollout} />
        ))}
      </div>
    </section>
  );
}

function VideoCard({ rollout }: { rollout: Rollout }) {
  const videoUrl = rollout.video_path ? `/api/videos/${rollout.video_path}` : null;

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow">
      {videoUrl ? (
        <video
          src={videoUrl}
          controls
          className="w-full h-48 object-cover bg-gray-100"
          preload="metadata"
        />
      ) : (
        <div className="w-full h-48 bg-gray-100 flex items-center justify-center">
          <Video className="h-12 w-12 text-gray-400" />
        </div>
      )}

      <div className="p-3 space-y-2">
        <h4 className="font-medium text-sm text-gray-900 line-clamp-2">
          {rollout.task_language_instruction || 'No task description'}
        </h4>
        <div className="space-y-1 text-xs text-gray-600">
          <p>
            <span className="font-medium">Dataset:</span> {rollout.dataset_name}
          </p>
          <p>
            <span className="font-medium">Robot:</span> {rollout.robot_embodiment}
          </p>
          {rollout.success_estimate !== null && rollout.success_estimate !== undefined && (
            <p>
              <span className="font-medium">Success:</span>{' '}
              <span
                className={
                  rollout.success_estimate > 0.7
                    ? 'text-green-600 font-semibold'
                    : rollout.success_estimate > 0.4
                      ? 'text-yellow-600 font-semibold'
                      : 'text-red-600 font-semibold'
                }
              >
                {(rollout.success_estimate * 100).toFixed(0)}%
              </span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
