'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import type { PlotParams } from 'react-plotly.js';

// Create a proper type for the Plot component
type PlotComponentType = React.ComponentType<PlotParams>;

// Dynamic import with proper typing
const PlotComponent = dynamic<PlotParams>(
  () => import('react-plotly.js').then((mod) => mod.default),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-96 flex items-center justify-center bg-gray-50 rounded-lg">
        <div className="text-gray-500">Loading chart...</div>
      </div>
    ),
  }
);

// Export wrapper component
export const PlotlyChart: React.FC<PlotParams> = (props) => {
  return <PlotComponent {...props} />;
};
