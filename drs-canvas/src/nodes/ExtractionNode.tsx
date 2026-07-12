import { Handle, Position } from '@xyflow/react';
import { Pickaxe } from 'lucide-react';

type VariableValue = number | string | { __type__: string; name: string };

interface VariableInfo {
  class: string;
  value: VariableValue;
}

interface ExtractionNodeData {
  label: string;
  class: string;
  layout?: { x: number; y: number };
  variables: {
    [key: string]: VariableInfo;
  };
  connections?: any;
  isContainer?: boolean;
}

export const ExtractionNode = ({ data, selected }: { data: ExtractionNodeData; selected?: boolean }) => {
  return (
    <div className={`px-4 py-3 shadow-xl rounded-xl border-2 bg-slate-900/95 backdrop-blur-md text-white min-w-[200px] transition-all duration-300 ${
      selected ? 'border-amber-500 shadow-amber-500/20' : 'border-amber-700/50 hover:border-amber-600'
    }`}>
      {/* Target handle: logical read (in case another block sets its rate/parameters) */}
      <Handle
        type="target"
        position={Position.Left}
        id="read-in"
        style={{ background: '#f59e0b', width: 10, height: 10, borderRadius: '2px', left: -6 }}
        title="Logical Read Input"
      />

      <div className="flex items-center gap-2 pb-2 mb-2 border-b border-amber-900/30">
        <Pickaxe className="w-5 h-5 text-amber-500" />
        <div>
          <div className="flex items-center gap-2">
            <div className="text-xs text-amber-500/80 font-bold uppercase tracking-wider">Mine Face</div>
            {data.isContainer && (
              <span className="text-[9px] bg-amber-950/60 border border-amber-800/40 px-1 py-0.5 rounded text-amber-300 font-semibold font-sans">
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
            <span className="font-semibold text-amber-400">{typeof varInfo.value === 'object' && varInfo.value !== null ? `${varInfo.value.__type__}: ${varInfo.value.name}` : varInfo.value}</span>
          </div>
        ))}
      </div>

      {/* Output handle: Physical flow of ore */}
      <Handle
        type="source"
        position={Position.Right}
        id="flow-out"
        style={{ background: '#3b82f6', width: 12, height: 12, border: '2px solid #1e293b', right: -6 }}
        title="Physical Flow Output"
      />
    </div>
  );
};
