'use client';

import { Database, Filter as FilterIcon } from 'lucide-react';

interface DashboardHeaderProps {
  totalRows: number;
  filteredRows: number;
}

export function DashboardHeader({ totalRows, filteredRows }: DashboardHeaderProps) {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <Database className="h-8 w-8 text-primary-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">ARES Dashboard</h1>
              <p className="text-sm text-gray-500">Automatic Robot Evaluation System</p>
            </div>
          </div>

          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2 text-sm">
              <FilterIcon className="h-4 w-4 text-gray-400" />
              <span className="text-gray-600">
                Showing <span className="font-semibold text-primary-600">{filteredRows}</span> of{' '}
                <span className="font-semibold">{totalRows}</span> rollouts
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
