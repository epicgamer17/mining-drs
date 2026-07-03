import { Handle, Position } from '@xyflow/react';
import { Factory } from 'lucide-react';

interface VariableInfo {
  class: string;
  value: number;
}

interface FactoryNodeData {
  label: string;
  class: string;
  layout?: { x: number; y: number };
  variables: {
    [key: string]: VariableInfo;
  };
}

export const FactoryNode = ({ data, selected }: { data: FactoryNodeData; selected?: boolean }) => {
  return (
    <div className={`px-4 py-3 shadow-xl rounded-xl border-2 bg-slate-900/95 backdrop-blur-md text-white min-w-[220px] transition-all duration-300 ${
      selected ? 'border-purple-500 shadow-purple-500/20' : 'border-purple-700/50 hover:border-purple-600'
    }`}>
      {/* Physical Flow Input */}
      <Handle
        type="target"
        position={Position.Left}
        id="flow-in"
        style={{ background: '#3b82f6', width: 12, height: 12, border: '2px solid #1e293b', left: -6 }}
        title="Physical Flow Input"
      />
      
      {/* Logical Read Input (amber) */}
      <Handle
        type="target"
        position={Position.Top}
        id="read-in"
        style={{ background: '#f59e0b', width: 10, height: 10, borderRadius: '2px', top: -6 }}
        title="Logical Read Input"
      />

      {/* Batch Data Stream Input (dashed green) */}
      <Handle
        type="target"
        position={Position.Bottom}
        id="data-in"
        style={{ background: '#10b981', width: 10, height: 10, borderRadius: '50%', bottom: -6 }}
        title="Conveyor Batch Data Input"
      />

      <div className="flex items-center gap-2 pb-2 mb-2 border-b border-purple-900/30">
        <Factory className="w-5 h-5 text-purple-400" />
        <div>
          <div className="text-xs text-purple-400 font-bold uppercase tracking-wider">Factory / Plant</div>
          <div className="text-sm font-semibold">{data.label}</div>
        </div>
      </div>

      <div className="space-y-1.5 text-xs text-slate-300">
        {Object.entries(data.variables || {}).map(([key, varInfo]) => (
          <div key={key} className="flex justify-between items-center gap-4 bg-slate-950/40 px-2 py-1 rounded">
            <span className="font-mono text-slate-400">{key}</span>
            <span className="font-semibold text-purple-300">{varInfo.value}</span>
          </div>
        ))}
      </div>

      {/* Physical Flow Output */}
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
