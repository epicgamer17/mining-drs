import { Handle, Position } from '@xyflow/react';
import { Database } from 'lucide-react';

type VariableValue = number | string | { __type__: string; name: string };

interface VariableInfo {
  class: string;
  value: VariableValue;
  lower_threshold?: number | string;
  upper_threshold?: number | string;
  rate?: number | string | { equation: string };
}

interface BufferNodeData {
  label: string;
  class: string;
  layout?: { x: number; y: number };
  variables: {
    [key: string]: VariableInfo;
  };
  isContainer?: boolean;
}

export const BufferNode = ({ data, selected }: { data: BufferNodeData; selected?: boolean }) => {
  // Find current mass/value to draw level bar
  let currentVal = 0.0;
  let upperLimit = 1000.0; // default cap for meter display
  
  // Try to find a Level variable to show a fill bar
  const levelVar = Object.values(data.variables || {}).find(v => v.class === 'Level' || v.class === 'Timer');
  if (levelVar) {
    currentVal = typeof levelVar.value === 'number' ? levelVar.value : 0.0;
    if (typeof levelVar.upper_threshold === 'number') {
      upperLimit = levelVar.upper_threshold;
    } else if (levelVar.upper_threshold === 'Infinity' || levelVar.upper_threshold === undefined) {
      upperLimit = Math.max(100.0, currentVal * 1.5);
    }
  }

  const fillPercent = Math.min(100, Math.max(0, (currentVal / (upperLimit || 1)) * 100));

  return (
    <div className={`px-4 py-3 shadow-xl rounded-xl border-2 bg-slate-900/95 backdrop-blur-md text-white min-w-[220px] transition-all duration-300 ${
      selected ? 'border-sky-500 shadow-sky-500/20' : 'border-sky-700/50 hover:border-sky-600'
    }`}>
      {/* Physical Flow Input */}
      <Handle
        type="target"
        position={Position.Left}
        id="flow-in"
        style={{ background: '#3b82f6', width: 12, height: 12, border: '2px solid #1e293b', left: -6 }}
        title="Physical Flow Input"
      />

      <div className="flex items-center gap-2 pb-2 mb-2 border-b border-sky-900/30">
        <Database className="w-5 h-5 text-sky-500" />
        <div>
          <div className="flex items-center gap-2">
            <div className="text-xs text-sky-500/80 font-bold uppercase tracking-wider">Buffer / Stockpile</div>
            {data.isContainer && (
              <span className="text-[9px] bg-sky-950/60 border border-sky-800/40 px-1 py-0.5 rounded text-sky-300 font-semibold font-sans">
                Sub-circuit
              </span>
            )}
          </div>
          <div className="text-sm font-semibold">{data.label}</div>
        </div>
      </div>

      {/* Visual Fill Bar */}
      {levelVar && (
        <div className="mb-3 bg-slate-950/60 p-2 rounded border border-sky-950/40">
          <div className="flex justify-between text-[10px] text-sky-400 font-mono mb-1">
            <span>Level</span>
            <span>{currentVal.toFixed(1)}t / {typeof levelVar.upper_threshold === 'number' ? levelVar.upper_threshold.toFixed(0) + 't' : '∞'}</span>
          </div>
          <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-sky-600 to-sky-400 rounded-full transition-all duration-500" 
              style={{ width: `${fillPercent}%` }}
            />
          </div>
        </div>
      )}

      <div className="space-y-1.5 text-xs text-slate-300">
        {Object.entries(data.variables || {}).map(([key, varInfo]) => (
          <div key={key} className="space-y-1 bg-slate-950/40 p-2 rounded">
            <div className="flex justify-between items-center">
              <span className="font-mono text-slate-400 font-medium">{key}</span>
              <span className="font-semibold text-sky-400">{typeof varInfo.value === 'object' && varInfo.value !== null ? (varInfo.value.__type__ ? `${varInfo.value.__type__}: ${varInfo.value.name}` : 'Expr') : varInfo.value}</span>
            </div>
            {varInfo.rate !== undefined && (
              <div className="flex justify-between items-center text-[10px] text-slate-500">
                <span>rate</span>
                <span 
                  className="text-amber-500/90 font-mono truncate max-w-[120px]" 
                  title={
                    typeof varInfo.rate === 'object' && varInfo.rate !== null && 'equation' in varInfo.rate
                      ? varInfo.rate.equation
                      : typeof varInfo.rate === 'string'
                      ? varInfo.rate
                      : String(varInfo.rate)
                  }
                >
                  {typeof varInfo.rate === 'object' && varInfo.rate !== null && 'equation' in varInfo.rate
                    ? varInfo.rate.equation
                    : typeof varInfo.rate === 'string'
                    ? varInfo.rate
                    : varInfo.rate}
                </span>
              </div>
            )}
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
      
      {/* Batch Data Stream Input (green, bottom-left) */}
      <Handle
        type="target"
        position={Position.Bottom}
        id="data-in"
        style={{ background: '#10b981', width: 10, height: 10, borderRadius: '50%', bottom: -6, left: -6 }}
        title="Batch Data Input"
      />

      {/* Logical Read Output (so other blocks can look up the level mass/volume) */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="read-out"
        style={{ background: '#f59e0b', width: 10, height: 10, borderRadius: '2px', bottom: -6 }}
        title="Logical Read Output"
      />
    </div>
  );
};
