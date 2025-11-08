'use client';

import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Database, Loader2 } from 'lucide-react';
import { DashboardHeader } from '@/components/layout/DashboardHeader';
import { StructuredFilters } from '@/components/filters/StructuredFilters';
import { EmbeddingFilters } from '@/components/filters/EmbeddingFilters';
import { DataDistributions } from '@/components/analytics/DataDistributions';
import { SuccessRateAnalytics } from '@/components/analytics/SuccessRateAnalytics';
import { TimeSeriesAnalytics } from '@/components/analytics/TimeSeriesAnalytics';
import { VideoGrid } from '@/components/display/VideoGrid';
import { HeroDisplay } from '@/components/display/HeroDisplay';
import { ExportOptions } from '@/components/export/ExportOptions';
import { useFilterStore } from '@/store/filterStore';
import { fetchRollouts } from '@/lib/api';

export default function Home() {
  const { filteredData, setData, applyFilters } = useFilterStore();

  const { data: rollouts, isLoading, error } = useQuery({
    queryKey: ['rollouts'],
    queryFn: fetchRollouts,
  });

  useEffect(() => {
    if (rollouts) {
      setData(rollouts);
      applyFilters();
    }
  }, [rollouts, setData, applyFilters]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin text-primary-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading ARES Dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <Database className="h-12 w-12 text-red-600 mx-auto mb-4" />
          <p className="text-red-600 font-semibold mb-2">Error loading data</p>
          <p className="text-gray-600">{(error as Error).message}</p>
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-gray-50">
      <DashboardHeader totalRows={rollouts?.length || 0} filteredRows={filteredData.length} />

      <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Data Filters */}
        <section className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Data Filters</h2>
          <StructuredFilters data={rollouts || []} />
        </section>

        {/* Embedding Filters */}
        <section className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Unstructured Data Filters</h2>
          <EmbeddingFilters />
        </section>

        <div className="h-px bg-gray-200" />

        {/* Data Sample */}
        <section className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Sample</h3>
          <p className="text-sm text-gray-600 mb-2">
            Showing {Math.min(5, filteredData.length)} of {filteredData.length} filtered rows
          </p>
        </section>

        {/* Analytics Sections */}
        <DataDistributions data={filteredData} />
        <SuccessRateAnalytics data={filteredData} />
        <TimeSeriesAnalytics data={filteredData} />

        <div className="h-px bg-gray-200" />

        {/* Video Grid */}
        <VideoGrid data={filteredData} />

        <div className="h-px bg-gray-200" />

        {/* Hero Display */}
        <HeroDisplay data={filteredData} />

        <div className="h-px bg-gray-200" />

        {/* Export Options */}
        <ExportOptions data={filteredData} />
      </div>
    </main>
  );
}
