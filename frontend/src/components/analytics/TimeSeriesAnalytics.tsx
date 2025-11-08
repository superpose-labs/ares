'use client';

import { useMemo } from 'react';
import { LineChart } from 'lucide-react';
import { Rollout } from '@/types';
import { format } from 'date-fns';
import { PlotlyChart } from '@/components/charts/PlotlyChart';

interface TimeSeriesAnalyticsProps {
  data: Rollout[];
}

const METRICS = [
  'success_estimate',
  'trajectory_length',
  'reward_percentage',
];

export function TimeSeriesAnalytics({ data }: TimeSeriesAnalyticsProps) {
  const visualizations = useMemo(() => {
    const dataWithDates = data
      .filter((row) => row.ingestion_time)
      .map((row) => ({
        ...row,
        date: new Date(row.ingestion_time),
      }))
      .sort((a, b) => a.date.getTime() - b.date.getTime());

    return METRICS.map((metric) => {
      const values = dataWithDates.map((row) => ({
        date: row.date,
        value: row[metric],
      })).filter((item) => item.value !== null && item.value !== undefined);

      // Group by date and calculate daily average
      const dailyData = values.reduce((acc: Record<string, number[]>, item) => {
        const dateStr = format(item.date, 'yyyy-MM-dd');
        if (!acc[dateStr]) acc[dateStr] = [];
        acc[dateStr].push(item.value);
        return acc;
      }, {});

      const dates = Object.keys(dailyData).sort();
      const avgValues = dates.map((date) => {
        const vals = dailyData[date];
        return vals.reduce((sum, v) => sum + v, 0) / vals.length;
      });

      return {
        title: metric.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
        data: [
          {
            x: dates,
            y: avgValues,
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            line: { color: '#0ea5e9', width: 2 },
            marker: { size: 6 },
          },
        ],
        layout: {
          title: `${metric.replace(/_/g, ' ')} Over Time`,
          xaxis: { title: 'Date' },
          yaxis: { title: metric.replace(/_/g, ' ') },
          height: 400,
        },
      };
    });
  }, [data]);

  return (
    <section className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center space-x-2 mb-6">
        <LineChart className="h-5 w-5 text-gray-500" />
        <h2 className="text-2xl font-bold text-gray-900">Time Series Trends</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
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
