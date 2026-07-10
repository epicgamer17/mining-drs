import React, { useState, useEffect } from 'react';
import { Sliders, Plus, Trash, Check, X } from 'lucide-react';

interface VariableInfo {
  class: string;
  value: number;
  lower_threshold?: number | string;
  upper_threshold?: number | string;
  rate?: number | string | { equation: string };
}

const isRateEquation = (rate: any): boolean => {
  if (typeof rate === 'object' && rate !== null && 'equation' in rate) return true;
  if (typeof rate === 'string' && rate !== 'Infinity' && rate !== '-Infinity' && rate !== 'NaN') return true;
  return false;
};

const getRateEquationString = (rate: any): string => {
  if (typeof rate === 'object' && rate !== null && 'equation' in rate) return rate.equation;
  if (typeof rate === 'string') return rate;
  return '';
};

interface NodeData {
  label: string;
  class: string;
  variables: {
    [key: string]: VariableInfo;
  };
  layout?: { x: number; y: number };
}

interface InspectorProps {
  selectedNode: {
    id: string;
    data: NodeData;
  } | null;
  onUpdateNode: (nodeId: string, updatedData: NodeData) => void;
  onClose: () => void;
}

export const Inspector = ({ selectedNode, onUpdateNode, onClose }: InspectorProps) => {
  const [label, setLabel] = useState('');
  const [nodeClass, setNodeClass] = useState('');
  const [variables, setVariables] = useState<{ [key: string]: VariableInfo }>({});
  const [newVarName, setNewVarName] = useState('');
  const [newVarClass, setNewVarClass] = useState('Variable');
  const [newVarValue, setNewVarValue] = useState('0.0');

  useEffect(() => {
    if (selectedNode) {
      setLabel(selectedNode.data.label);
      setNodeClass(selectedNode.data.class);
      setVariables(JSON.parse(JSON.stringify(selectedNode.data.variables || {})));
    }
  }, [selectedNode]);

  if (!selectedNode) return null;

  const handleSave = () => {
    // Basic verification of validation boundaries
    for (const [name, varInfo] of Object.entries(variables)) {
      const val = varInfo.value;
      const lower = typeof varInfo.lower_threshold === 'number' ? varInfo.lower_threshold : -Infinity;
      const upper = typeof varInfo.upper_threshold === 'number' ? varInfo.upper_threshold : Infinity;
      if (val < lower || val > upper) {
        alert(`Validation Warning: Variable "${name}" value (${val}) violates thresholds [${lower}, ${upper}].`);
      }
    }

    onUpdateNode(selectedNode.id, {
      ...selectedNode.data,
      label,
      class: nodeClass,
      variables,
    });
  };

  const handleAddVariable = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newVarName.trim()) return;
    if (variables[newVarName]) {
      alert('Variable name already exists.');
      return;
    }

    const valueNum = parseFloat(newVarValue) || 0;
    const updated = {
      ...variables,
      [newVarName]: {
        class: newVarClass,
        value: valueNum,
        ...(newVarClass === 'Level' || newVarClass === 'Timer' ? {
          rate: 0.0,
          lower_threshold: '-Infinity',
          upper_threshold: 'Infinity'
        } : {})
      }
    };
    setVariables(updated);
    setNewVarName('');
    setNewVarValue('0.0');
  };

  const handleRemoveVariable = (nameToRemove: string) => {
    const updated = { ...variables };
    delete updated[nameToRemove];
    setVariables(updated);
  };

  const handleVariableChange = (name: string, field: keyof VariableInfo, val: string, isEquation: boolean = false) => {
    const updated = { ...variables };
    const currentVar = updated[name];
    if (!currentVar) return;

    if (field === 'rate') {
      if (isEquation) {
        currentVar.rate = { equation: val };
      } else {
        currentVar.rate = parseFloat(val) || 0;
      }
    } else if (field === 'value') {
      currentVar[field] = parseFloat(val) || 0;
    } else if (field === 'lower_threshold' || field === 'upper_threshold') {
      if (val === '-Infinity' || val === 'Infinity' || val === '') {
        currentVar[field] = val;
      } else {
        currentVar[field] = parseFloat(val) || 0;
      }
    } else if (field === 'class') {
      currentVar[field] = val;
    }
    setVariables(updated);
  };

  return (
    <aside className="w-80 bg-slate-900 border-l border-slate-800 p-4 flex flex-col justify-between text-white overflow-y-auto">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-sky-400" />
            <h2 className="text-sm font-bold">Node Inspector</h2>
          </div>
          <button onClick={onClose} className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Global Node Config */}
        <div className="space-y-3">
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Dotted Path ID</label>
            <input
              type="text"
              value={selectedNode.id}
              disabled
              className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-slate-400 font-mono"
            />
          </div>
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Label Name</label>
            <input
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              className="w-full bg-slate-950 border border-slate-850 rounded px-2 py-1.5 text-xs text-white focus:border-sky-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Backend Class</label>
            <input
              type="text"
              value={nodeClass}
              onChange={(e) => setNodeClass(e.target.value)}
              className="w-full bg-slate-950 border border-slate-850 rounded px-2 py-1.5 text-xs text-white focus:border-sky-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Variables Section */}
        <div className="space-y-3">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Registered Variables</h3>
          
          <div className="space-y-4 max-h-[350px] overflow-y-auto pr-1">
            {Object.entries(variables).map(([name, varInfo]) => (
              <div key={name} className="p-2 rounded border border-slate-800 bg-slate-950/60 space-y-2 text-xs relative group">
                <button
                  onClick={() => handleRemoveVariable(name)}
                  className="absolute top-2 right-2 text-red-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <Trash className="w-3.5 h-3.5" />
                </button>

                <div className="font-mono text-slate-200 font-bold pr-5">{name}</div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-[9px] text-slate-500 uppercase tracking-wider">Class</span>
                    <select
                      value={varInfo.class}
                      onChange={(e) => handleVariableChange(name, 'class', e.target.value)}
                      className="w-full bg-slate-900 border border-slate-800 rounded px-1 py-0.5 mt-0.5"
                    >
                      <option value="Variable">Variable</option>
                      <option value="Level">Level</option>
                      <option value="Timer">Timer</option>
                    </select>
                  </div>
                  <div>
                    <span className="text-[9px] text-slate-500 uppercase tracking-wider">Value</span>
                    <input
                      type="number"
                      step="any"
                      value={varInfo.value}
                      onChange={(e) => handleVariableChange(name, 'value', e.target.value)}
                      className="w-full bg-slate-900 border border-slate-800 rounded px-1 py-0.5 mt-0.5 font-mono text-sky-400"
                    />
                  </div>
                </div>

                {(varInfo.class === 'Level' || varInfo.class === 'Timer') && (
                  <div className="space-y-2 pt-1 border-t border-slate-900">
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase tracking-wider">Min Bound</span>
                        <input
                          type="text"
                          value={varInfo.lower_threshold ?? '-Infinity'}
                          onChange={(e) => handleVariableChange(name, 'lower_threshold', e.target.value)}
                          className="w-full bg-slate-900 border border-slate-800 rounded px-1.5 py-0.5 mt-0.5 font-mono"
                        />
                      </div>
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase tracking-wider">Max Bound</span>
                        <input
                          type="text"
                          value={varInfo.upper_threshold ?? 'Infinity'}
                          onChange={(e) => handleVariableChange(name, 'upper_threshold', e.target.value)}
                          className="w-full bg-slate-900 border border-slate-800 rounded px-1.5 py-0.5 mt-0.5 font-mono"
                        />
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between items-center mb-0.5">
                        <span className="text-[9px] text-slate-500 uppercase tracking-wider">Current Rate</span>
                        <div className="flex gap-1 bg-slate-900 p-0.5 rounded border border-slate-800">
                          <button
                            type="button"
                            onClick={() => {
                              const hasEq = isRateEquation(varInfo.rate);
                              const currentVal = hasEq ? 0.0 : (parseFloat(varInfo.rate as string) || 0.0);
                              handleVariableChange(name, 'rate', currentVal.toString(), false);
                            }}
                            className={`px-1.5 py-0.5 text-[9px] rounded font-semibold transition-all ${
                              !isRateEquation(varInfo.rate) ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200'
                            }`}
                          >
                            Static
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              const hasEq = isRateEquation(varInfo.rate);
                              const defaultEq = hasEq ? getRateEquationString(varInfo.rate) : '';
                              handleVariableChange(name, 'rate', defaultEq, true);
                            }}
                            className={`px-1.5 py-0.5 text-[9px] rounded font-semibold transition-all ${
                              isRateEquation(varInfo.rate) ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-slate-200'
                            }`}
                          >
                            Equation
                          </button>
                        </div>
                      </div>
                      {isRateEquation(varInfo.rate) ? (
                        <input
                          type="text"
                          value={getRateEquationString(varInfo.rate)}
                          onChange={(e) => handleVariableChange(name, 'rate', e.target.value, true)}
                          placeholder="e.g. self.sibling.mass * 0.1"
                          className="w-full bg-slate-900 border border-slate-800 rounded px-1.5 py-0.5 mt-0.5 font-mono text-amber-400 text-xs focus:border-amber-500 focus:outline-none"
                        />
                      ) : (
                        <input
                          type="number"
                          step="any"
                          value={typeof varInfo.rate === 'number' ? varInfo.rate : parseFloat(varInfo.rate as string) || 0.0}
                          onChange={(e) => handleVariableChange(name, 'rate', e.target.value, false)}
                          className="w-full bg-slate-900 border border-slate-800 rounded px-1.5 py-0.5 mt-0.5 font-mono text-sky-400"
                        />
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Add Variable Form */}
        <form onSubmit={handleAddVariable} className="p-2 border border-slate-800 rounded-lg bg-slate-950/20 space-y-2.5">
          <span className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider">Add Variable</span>
          <div className="grid grid-cols-2 gap-2">
            <input
              type="text"
              placeholder="Name"
              value={newVarName}
              onChange={(e) => setNewVarName(e.target.value)}
              className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs"
            />
            <select
              value={newVarClass}
              onChange={(e) => setNewVarClass(e.target.value)}
              className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs"
            >
              <option value="Variable">Variable</option>
              <option value="Level">Level</option>
              <option value="Timer">Timer</option>
            </select>
          </div>
          <div className="flex gap-2">
            <input
              type="number"
              placeholder="Val"
              value={newVarValue}
              onChange={(e) => setNewVarValue(e.target.value)}
              className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs w-2/3 font-mono"
            />
            <button
              type="submit"
              className="flex-1 flex items-center justify-center gap-1 bg-sky-600 hover:bg-sky-500 rounded text-xs font-semibold"
            >
              <Plus className="w-3.5 h-3.5" />
              Add
            </button>
          </div>
        </form>
      </div>

      {/* Save Button */}
      <div className="pt-4 border-t border-slate-800 flex gap-2">
        <button
          onClick={handleSave}
          className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-gradient-to-r from-sky-600 to-sky-500 hover:from-sky-500 hover:to-sky-400 rounded-lg text-xs font-bold shadow-lg shadow-sky-950/45 transition-all"
        >
          <Check className="w-4 h-4" />
          Apply Changes
        </button>
      </div>
    </aside>
  );
};
