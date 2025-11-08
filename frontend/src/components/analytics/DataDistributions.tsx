'use client';

import { useState, useMemo } from 'react';
import { BarChart3 } from 'lucide-react';
import { Rollout } from '@/types';
import { PlotlyChart } from '@/components/charts/PlotlyChart';

interface DataDistributionsProps {
  data: Rollout[];
}

const NUMERIC_COLUMNS = [
  'success_estimate',
  'trajectory_length',
  'reward_step',
  'reward_percentage',
];

const CATEGORICAL_COLUMNS = [
  'dataset_name',
  'robot_embodiment',
  'robot_color_estimate',
  'environment_surface_estimate',
  'environment_lighting_estimate',
];

export function DataDistributions({ data }: DataDistributionsProps) {
  const [activeTab, setActiveTab] = useState(0);

  const visualizations = useMemo(() => {
    const vizs: any[] = [];

    // Histograms for numeric columns
    NUMERIC_COLUMNS.forEach((col) => {
      const values = data.map((row) => row[col]).filter((v) => v !== null && v !== undefined);
      if (values.length > 0) {
        vizs.push({
          title: col.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
          type: 'histogram',
          data: [
            {
              x: values,
              type: 'histogram',
              marker: { color: '#0ea5e9' },
              nbinsx: 30,
            },
          ],
          layout: {
            title: `Distribution of ${col.replace(/_/g, ' ')}`,
            xaxis: { title: col.replace(/_/g, ' ') },
            yaxis: { title: 'Count' },
            height: 400,
          },
        });
      }
    });

    // Bar charts for categorical columns
    CATEGORICAL_COLUMNS.forEach((col) => {
      const values = data.map((row) => row[col] || '(None)');
      const counts = values.reduce((acc: Record<string, number>, val) => {
        acc[val] = (acc[val] || 0) + 1;
        return acc;
      }, {});

      const sortedEntries = Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20);

      if (sortedEntries.length > 0) {
        vizs.push({
          title: col.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
          type: 'bar',
          data: [
            {
              x: sortedEntries.map(([k]) => k),
              y: sortedEntries.map(([, v]) => v),
              type: 'bar',
              marker: { color: '#10b981' },
            },
          ],
          layout: {
            title: `Distribution of ${col.replace(/_/g, ' ')}`,
            xaxis: { title: col.replace(/_/g, ' ') },
            yaxis: { title: 'Count' },
            height: 400,
          },
        });
      }
    });

    return vizs;
  }, [data]);

  if (visualizations.length === 0) {
    return null;
  }

  return (
    <section className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center space-x-2 mb-6">
        <BarChart3 className="h-5 w-5 text-gray-500" />
        <h2 className="text-2xl font-bold text-gray-900">Distribution Analytics</h2>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <div className="flex space-x-2 overflow-x-auto">
          {visualizations.map((viz, idx) => (
            <button
              key={idx}
              onClick={() => setActiveTab(idx)}
              className={`px-4 py-2 font-medium text-sm whitespace-nowrap transition-colors ${
                activeTab === idx
                  ? 'border-b-2 border-primary-600 text-primary-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {viz.title}
            </button>
          ))}
        </div>
      </div>

      {/* Visualization */}
      {visualizations[activeTab] && (
        <PlotlyChart
          data={visualizations[activeTab].data}
          layout={{
            ...visualizations[activeTab].layout,
            autosize: true,
          }}
          config={{ displayModeBar: true, displaylogo: false }}
          className="w-full"
          useResizeHandler
        />
      )}
    </section>
  );
}
