'use client';

import { useState, useMemo } from 'react';
import { Sliders, X, RefreshCw } from 'lucide-react';
import { useFilterStore } from '@/store/filterStore';
import { Rollout, Filter } from '@/types';
import { getUniqueValues, getNumericRange } from '@/lib/utils';

interface StructuredFiltersProps {
  data: Rollout[];
}

const NUMERIC_FIELDS = [
  'success_estimate',
  'trajectory_length',
  'reward_step',
  'reward_percentage',
];

const CATEGORICAL_FIELDS = [
  'dataset_name',
  'robot_embodiment',
  'robot_color_estimate',
  'environment_surface_estimate',
  'environment_lighting_estimate',
  'task_type',
];

export function StructuredFilters({ data }: StructuredFiltersProps) {
  const { structuredFilters, addStructuredFilter, removeStructuredFilter, clearStructuredFilters } =
    useFilterStore();

  const [tempFilters, setTempFilters] = useState<Record<string, any>>({});

  const handleNumericFilter = (field: string, min: number, max: number, includeNaN: boolean) => {
    setTempFilters((prev) => ({ ...prev, [field]: { min, max, includeNaN } }));
  };

  const handleCategoricalFilter = (field: string, values: string[]) => {
    setTempFilters((prev) => ({ ...prev, [field]: values }));
  };

  const applyTempFilters = () => {
    Object.entries(tempFilters).forEach(([field, value]) => {
      if (NUMERIC_FIELDS.includes(field)) {
        const { min, max, includeNaN } = value;
        addStructuredFilter({
          type: 'range',
          field,
          value: [min, max],
          includeNaN,
        });
      } else if (value.length > 0) {
        addStructuredFilter({
          type: 'categorical',
          field,
          value,
        });
      }
    });
    setTempFilters({});
  };

  const resetFilters = () => {
    clearStructuredFilters();
    setTempFilters({});
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Sliders className="h-5 w-5 text-gray-500" />
          <h3 className="text-lg font-semibold text-gray-900">Structured Filters</h3>
        </div>
        <button
          onClick={resetFilters}
          className="flex items-center space-x-1 px-3 py-1 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors"
        >
          <RefreshCw className="h-4 w-4" />
          <span>Reset All</span>
        </button>
      </div>

      {/* Active Filters */}
      {structuredFilters.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {structuredFilters.map((filter) => (
            <div
              key={filter.field}
              className="flex items-center space-x-1 px-3 py-1 bg-primary-100 text-primary-800 rounded-full text-sm"
            >
              <span className="font-medium">{filter.field.replace(/_/g, ' ')}</span>
              <span>:</span>
              <span>
                {filter.type === 'range'
                  ? `${(filter.value as [number, number])[0].toFixed(2)} - ${(filter.value as [number, number])[1].toFixed(2)}`
                  : `${(filter.value as string[]).length} selected`}
              </span>
              <button
                onClick={() => removeStructuredFilter(filter.field)}
                className="ml-1 hover:bg-primary-200 rounded-full p-0.5"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Numeric Filters */}
      <div className="space-y-4">
        <h4 className="font-medium text-gray-700">Numeric Filters</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {NUMERIC_FIELDS.map((field) => (
            <NumericFilter
              key={field}
              field={field}
              data={data}
              onChange={handleNumericFilter}
            />
          ))}
        </div>
      </div>

      {/* Categorical Filters */}
      <div className="space-y-4">
        <h4 className="font-medium text-gray-700">Categorical Filters</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {CATEGORICAL_FIELDS.map((field) => (
            <CategoricalFilter
              key={field}
              field={field}
              data={data}
              onChange={handleCategoricalFilter}
            />
          ))}
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={applyTempFilters}
          className="px-6 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors font-medium"
        >
          Apply Filters
        </button>
      </div>
    </div>
  );
}

function NumericFilter({
  field,
  data,
  onChange,
}: {
  field: string;
  data: Rollout[];
  onChange: (field: string, min: number, max: number, includeNaN: boolean) => void;
}) {
  const [min, max] = useMemo(() => getNumericRange(data, field), [data, field]);
  const [localMin, setLocalMin] = useState(min);
  const [localMax, setLocalMax] = useState(max);
  const [includeNaN, setIncludeNaN] = useState(false);

  return (
    <div className="border border-gray-200 rounded-lg p-4 space-y-3">
      <label className="block text-sm font-medium text-gray-700">
        {field.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
      </label>
      <div className="space-y-2">
        <input
          type="range"
          min={min}
          max={max}
          step={(max - min) / 100}
          value={localMin}
          onChange={(e) => {
            const val = parseFloat(e.target.value);
            setLocalMin(val);
            onChange(field, val, localMax, includeNaN);
          }}
          className="w-full"
        />
        <input
          type="range"
          min={min}
          max={max}
          step={(max - min) / 100}
          value={localMax}
          onChange={(e) => {
            const val = parseFloat(e.target.value);
            setLocalMax(val);
            onChange(field, localMin, val, includeNaN);
          }}
          className="w-full"
        />
      </div>
      <div className="flex justify-between text-xs text-gray-600">
        <span>{localMin.toFixed(2)}</span>
        <span>{localMax.toFixed(2)}</span>
      </div>
      <label className="flex items-center space-x-2 text-sm">
        <input
          type="checkbox"
          checked={includeNaN}
          onChange={(e) => {
            setIncludeNaN(e.target.checked);
            onChange(field, localMin, localMax, e.target.checked);
          }}
          className="rounded border-gray-300"
        />
        <span className="text-gray-600">Include NaN values</span>
      </label>
    </div>
  );
}

function CategoricalFilter({
  field,
  data,
  onChange,
}: {
  field: string;
  data: Rollout[];
  onChange: (field: string, values: string[]) => void;
}) {
  const options = useMemo(() => getUniqueValues(data, field).slice(0, 25), [data, field]);
  const [selected, setSelected] = useState<string[]>([]);

  const handleToggle = (value: string) => {
    const newSelected = selected.includes(value)
      ? selected.filter((v) => v !== value)
      : [...selected, value];
    setSelected(newSelected);
    onChange(field, newSelected);
  };

  return (
    <div className="border border-gray-200 rounded-lg p-4 space-y-3">
      <label className="block text-sm font-medium text-gray-700">
        {field.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
      </label>
      <div className="max-h-40 overflow-y-auto space-y-1">
        {options.map((option) => (
          <label key={String(option)} className="flex items-center space-x-2 text-sm">
            <input
              type="checkbox"
              checked={selected.includes(String(option))}
              onChange={() => handleToggle(String(option))}
              className="rounded border-gray-300"
            />
            <span className="text-gray-600">{String(option)}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
