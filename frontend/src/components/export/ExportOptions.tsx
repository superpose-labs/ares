'use client';

import { useState } from 'react';
import { Download } from 'lucide-react';
import { Rollout } from '@/types';
import { exportData } from '@/lib/api';
import { downloadBlob } from '@/lib/utils';
import Papa from 'papaparse';
import jsPDF from 'jspdf';

interface ExportOptionsProps {
  data: Rollout[];
}

export function ExportOptions({ data }: ExportOptionsProps) {
  const [exporting, setExporting] = useState(false);

  const handleExport = async (format: string) => {
    setExporting(true);
    try {
      if (format === 'csv') {
        const csv = Papa.unparse(data);
        const blob = new Blob([csv], { type: 'text/csv' });
        downloadBlob(blob, `ares_export_${Date.now()}.csv`);
      } else if (format === 'json') {
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        downloadBlob(blob, `ares_export_${Date.now()}.json`);
      } else if (format === 'pdf') {
        const doc = new jsPDF();
        doc.text('ARES Dashboard Export', 10, 10);
        doc.text(`Total Rollouts: ${data.length}`, 10, 20);
        doc.text(`Export Date: ${new Date().toISOString()}`, 10, 30);
        doc.save(`ares_export_${Date.now()}.pdf`);
      } else {
        // For server-side formats (parquet, etc.)
        const blob = await exportData(data, format);
        downloadBlob(blob, `ares_export_${Date.now()}.${format}`);
      }
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed. Please try again.');
    } finally {
      setExporting(false);
    }
  };

  return (
    <section className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center space-x-2 mb-6">
        <Download className="h-5 w-5 text-gray-500" />
        <h2 className="text-2xl font-bold text-gray-900">Export Data</h2>
      </div>

      <div className="space-y-4">
        <p className="text-gray-600">
          Export {data.length} filtered rollouts in your preferred format
        </p>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          <ExportButton
            label="CSV"
            description="Comma-separated values"
            onClick={() => handleExport('csv')}
            disabled={exporting}
          />
          <ExportButton
            label="JSON"
            description="JavaScript Object Notation"
            onClick={() => handleExport('json')}
            disabled={exporting}
          />
          <ExportButton
            label="PDF"
            description="Portable Document Format"
            onClick={() => handleExport('pdf')}
            disabled={exporting}
          />
          <ExportButton
            label="Parquet"
            description="Columnar storage format"
            onClick={() => handleExport('parquet')}
            disabled={exporting}
          />
          <ExportButton
            label="HTML"
            description="Web page format"
            onClick={() => handleExport('html')}
            disabled={exporting}
          />
        </div>
      </div>
    </section>
  );
}

function ExportButton({
  label,
  description,
  onClick,
  disabled,
}: {
  label: string;
  description: string;
  onClick: () => void;
  disabled: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="border-2 border-gray-200 rounded-lg p-4 hover:border-primary-500 hover:bg-primary-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-left"
    >
      <div className="flex items-center justify-center mb-2">
        <Download className="h-8 w-8 text-primary-600" />
      </div>
      <h3 className="font-semibold text-gray-900 text-center">{label}</h3>
      <p className="text-xs text-gray-500 text-center mt-1">{description}</p>
    </button>
  );
}
