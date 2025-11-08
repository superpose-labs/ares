import { create } from 'zustand';
import { Rollout, Filter, EmbeddingSelection } from '@/types';

interface FilterState {
  data: Rollout[];
  filteredData: Rollout[];
  structuredFilters: Filter[];
  embeddingSelections: EmbeddingSelection[];
  selectedRow: Rollout | null;

  setData: (data: Rollout[]) => void;
  addStructuredFilter: (filter: Filter) => void;
  removeStructuredFilter: (field: string) => void;
  clearStructuredFilters: () => void;
  addEmbeddingSelection: (selection: EmbeddingSelection) => void;
  clearEmbeddingSelections: () => void;
  setSelectedRow: (row: Rollout | null) => void;
  applyFilters: () => void;
}

export const useFilterStore = create<FilterState>((set, get) => ({
  data: [],
  filteredData: [],
  structuredFilters: [],
  embeddingSelections: [],
  selectedRow: null,

  setData: (data) => set({ data, filteredData: data }),

  addStructuredFilter: (filter) => {
    const filters = get().structuredFilters.filter((f) => f.field !== filter.field);
    set({ structuredFilters: [...filters, filter] });
    get().applyFilters();
  },

  removeStructuredFilter: (field) => {
    set({ structuredFilters: get().structuredFilters.filter((f) => f.field !== field) });
    get().applyFilters();
  },

  clearStructuredFilters: () => {
    set({ structuredFilters: [] });
    get().applyFilters();
  },

  addEmbeddingSelection: (selection) => {
    const selections = get().embeddingSelections.filter((s) => s.type !== selection.type);
    set({ embeddingSelections: [...selections, selection] });
    get().applyFilters();
  },

  clearEmbeddingSelections: () => {
    set({ embeddingSelections: [] });
    get().applyFilters();
  },

  setSelectedRow: (row) => set({ selectedRow: row }),

  applyFilters: () => {
    const { data, structuredFilters, embeddingSelections } = get();
    let filtered = [...data];

    // Apply structured filters
    structuredFilters.forEach((filter) => {
      if (filter.type === 'range') {
        const [min, max] = filter.value as [number, number];
        filtered = filtered.filter((row) => {
          const value = row[filter.field];
          if (value === null || value === undefined) {
            return filter.includeNaN || false;
          }
          return value >= min && value <= max;
        });
      } else if (filter.type === 'categorical') {
        const values = filter.value as string[];
        filtered = filtered.filter((row) => {
          const value = row[filter.field];
          const strValue = value === null || value === undefined ? '(None)' : String(value);
          return values.includes(strValue);
        });
      }
    });

    // Apply embedding selections (AND operation)
    if (embeddingSelections.length > 0) {
      const allSelectedIds = embeddingSelections.reduce<Set<string>>((acc, selection, index) => {
        if (index === 0) {
          return new Set(selection.selectedIds);
        }
        const intersection = new Set<string>();
        selection.selectedIds.forEach((id) => {
          if (acc.has(id)) intersection.add(id);
        });
        return intersection;
      }, new Set());

      filtered = filtered.filter((row) => allSelectedIds.has(row.id));
    }

    set({ filteredData: filtered });
  },
}));
