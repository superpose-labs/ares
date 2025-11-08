'use client';

import { useMemo } from 'react';
import { TrendingUp } from 'lucide-react';
import { Rollout } from '@/types';
import { PlotlyChart } from '@/components/charts/PlotlyChart';

interface SuccessRateAnalyticsProps {
  data: Rollout[];
}

const GROUPING_COLUMNS = [
  'dataset_name',
  'robot_embodiment',
  'environment_surface_estimate',
  'environment_lighting_estimate',
];

export function SuccessRateAnalytics({ data }: SuccessRateAnalyticsProps) {
  const visualizations = useMemo(() => {
    return GROUPING_COLUMNS.map((col) => {
      const groups = data.reduce((acc: Record<string, number[]>, row) => {
        const key = row[col] || '(None)';
        const success = row.success_estimate;
        if (success !== null && success !== undefined) {
          if (!acc[key]) acc[key] = [];
          acc[key].push(success);
        }
        return acc;
      }, {});

      const avgSuccessRates = Object.entries(groups)
        .map(([key, values]) => ({
          key,
          avg: values.reduce((sum, v) => sum + v, 0) / values.length,
          count: values.length,
        }))
        .sort((a, b) => b.avg - a.avg)
        .slice(0, 15);

      return {
        title: col.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
        data: [
          {
            x: avgSuccessRates.map((item) => item.key),
            y: avgSuccessRates.map((item) => item.avg),
            type: 'bar' as const,
            marker: { color: '#10b981' },
            text: avgSuccessRates.map((item) => `n=${item.count}`),
            textposition: 'auto' as const,
          },
        ],
        layout: {
          title: `Average Success Rate by ${col.replace(/_/g, ' ')}`,
          xaxis: { title: col.replace(/_/g, ' ') },
          yaxis: { title: 'Average Success Rate', range: [0, 1] },
          height: 400,
        },
      };
    });
  }, [data]);

  return (
    <section className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center space-x-2 mb-6">
        <TrendingUp className="h-5 w-5 text-gray-500" />
        <h2 className="text-2xl font-bold text-gray-900">Success Rate Analytics</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {visualizations.map((viz, idx) => (
          <div key={idx} className="border border-gray-200 rounded-lg p-4">
            <PlotlyChart
              data={viz.data}
              layout={{ ...viz.layout, autosize: true }}
              config={{ displayModeBar: true, displaylogo: false }}
              className="w-full"
              useResizeHandler
            />
          </div>
        ))}
      </div>
    </section>
  );
}
