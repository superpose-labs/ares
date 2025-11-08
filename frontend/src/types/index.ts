export interface Rollout {
  id: string;
  dataset_name: string;
  dataset_formalname: string;
  task_language_instruction: string;
  task_type?: string;
  success_estimate?: number;
  robot_embodiment: string;
  robot_color_estimate?: string;
  environment_surface_estimate?: string;
  environment_lighting_estimate?: string;
  trajectory_length?: number;
  ingestion_time?: string;
  video_path: string;
  [key: string]: any;
}

export interface Filter {
  type: 'range' | 'categorical';
  field: string;
  value: [number, number] | string[];
  includeNaN?: boolean;
}

export interface EmbeddingSelection {
  type: 'task' | 'description' | 'trajectory';
  selectedIds: string[];
}

export interface VisualizationData {
  title: string;
  figure: any; // Plotly figure data
  type: 'histogram' | 'bar' | 'line' | 'scatter';
}
