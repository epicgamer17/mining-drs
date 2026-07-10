import { Handle, Position } from '@xyflow/react';
import { RefreshCw } from 'lucide-react';

interface VariableInfo {
  class: string;
  value: number;
}

interface DataSourceNodeData {
  label: string;
  class: string;
  layout?: { x: number; y: number };
  variables: {
    [key: string]: VariableInfo;
  };
  isContainer?: boolean;
}

export const DataSourceNode = ({ data, selected }: { data: DataSourceNodeData; selected?: boolean }) => {
  return (
    <div className={`px-4 py-3 shadow-xl rounded-xl border-2 bg-slate-900/95 backdrop-blur-md text-white min-w-[200px] transition-all duration-300 ${
      selected ? 'border-emerald-500 shadow-emerald-500/20' : 'border-emerald-700/50 hover:border-emerald-600'
    }`}>
      <div className="flex items-center gap-2 pb-2 mb-2 border-b border-emerald-900/30">
        <RefreshCw className="w-5 h-5 text-emerald-500" />
        <div>
          <div className="flex items-center gap-2">
            <div className="text-xs text-emerald-500/80 font-bold uppercase tracking-wider">Data Source</div>
            {data.isContainer && (
              <span className="text-[9px] bg-emerald-950/60 border border-emerald-800/40 px-1 py-0.5 rounded text-emerald-300 font-semibold font-sans">
                Sub-circuit
              </span>
            )}
          </div>
          <div className="text-sm font-semibold">{data.label}</div>
        </div>
      </div>

      <div className="space-y-1.5 text-xs text-slate-300">
        {Object.entries(data.variables || {}).map(([key, varInfo]) => (
          <div key={key} className="flex justify-between items-center gap-4 bg-slate-950/40 px-2 py-1 rounded">
            <span className="font-mono text-slate-400">{key}</span>
            <span className="font-semibold text-emerald-400">{varInfo.value}</span>
          </div>
        ))}
      </div>

      {/* Batch Data Output handle (green) */}
      <Handle
        type="source"
        position={Position.Right}
        id="data-out"
        style={{ background: '#10b981', width: 10, height: 10, borderRadius: '50%', right: -6 }}
        title="Batch Data Stream Output"
      />
    </div>
  );
};
