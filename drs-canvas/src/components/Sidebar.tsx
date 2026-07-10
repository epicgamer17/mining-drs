import React from 'react';
import { Pickaxe, Database, Factory, RefreshCw, Download, Upload, Trash2, Layers, Save, CheckCircle } from 'lucide-react';

interface SidebarProps {
  onExport: (format: 'flat' | 'hierarchical') => void;
  onImport: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onClear: () => void;
  onSaveToWorkspace: () => void;
  onVerifyCompile: () => void;
  isSaving: boolean;
  isCompiling: boolean;
}

export const Sidebar = ({ 
  onExport, 
  onImport, 
  onClear,
  onSaveToWorkspace,
  onVerifyCompile,
  isSaving,
  isCompiling
}: SidebarProps) => {
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const triggerImportClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <aside className="w-80 bg-slate-900 border-r border-slate-800 p-4 flex flex-col justify-between text-white select-none">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center gap-2 pb-4 border-b border-slate-800">
          <Layers className="w-6 h-6 text-sky-400" />
          <div>
            <h1 className="text-md font-bold tracking-tight">DRS Canvas Editor</h1>
            <p className="text-[10px] text-slate-400">Drag & Drop Module Topology</p>
          </div>
        </div>

        {/* Draggable Components */}
        <div className="space-y-3">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Module Palette</h2>
          
          <div
            className="flex items-center gap-3 p-3 rounded-lg border border-slate-800 bg-slate-950/40 hover:border-amber-700/60 hover:bg-amber-950/10 cursor-grab active:cursor-grabbing transition-all"
            draggable
            onDragStart={(e) => onDragStart(e, 'extractionNode')}
          >
            <Pickaxe className="w-5 h-5 text-amber-500" />
            <div>
              <div className="text-xs font-semibold">Mine Face</div>
              <div className="text-[10px] text-slate-500">Resource extraction block</div>
            </div>
          </div>

          <div
            className="flex items-center gap-3 p-3 rounded-lg border border-slate-800 bg-slate-950/40 hover:border-sky-700/60 hover:bg-sky-950/10 cursor-grab active:cursor-grabbing transition-all"
            draggable
            onDragStart={(e) => onDragStart(e, 'bufferNode')}
          >
            <Database className="w-5 h-5 text-sky-500" />
            <div>
              <div className="text-xs font-semibold">Stockpile / Crusher</div>
              <div className="text-[10px] text-slate-500">Storage and level buffer block</div>
            </div>
          </div>

          <div
            className="flex items-center gap-3 p-3 rounded-lg border border-slate-800 bg-slate-950/40 hover:border-purple-700/60 hover:bg-purple-950/10 cursor-grab active:cursor-grabbing transition-all"
            draggable
            onDragStart={(e) => onDragStart(e, 'factoryNode')}
          >
            <Factory className="w-5 h-5 text-purple-400" />
            <div>
              <div className="text-xs font-semibold">Processing Plant</div>
              <div className="text-[10px] text-slate-500">Continuous refiner and factory</div>
            </div>
          </div>

          <div
            className="flex items-center gap-3 p-3 rounded-lg border border-slate-800 bg-slate-950/40 hover:border-emerald-700/60 hover:bg-emerald-950/10 cursor-grab active:cursor-grabbing transition-all"
            draggable
            onDragStart={(e) => onDragStart(e, 'dataSourceNode')}
          >
            <RefreshCw className="w-5 h-5 text-emerald-500" />
            <div>
              <div className="text-xs font-semibold">Data Source</div>
              <div className="text-[10px] text-slate-500">Batch conveyer stream feed</div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Controls & Save/Load */}
      <div className="space-y-3 pt-4 border-t border-slate-800">
        <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Topology Pipeline</h2>
        
        {/* Workspace Sync */}
        <div className="space-y-2 pb-2 border-b border-slate-800/60">
          <button
            onClick={onSaveToWorkspace}
            disabled={isSaving}
            className={`w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all ${
              isSaving 
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-800' 
                : 'bg-gradient-to-r from-sky-600 to-sky-500 hover:from-sky-500 hover:to-sky-400 text-white shadow-lg shadow-sky-950/20 border border-sky-600/30'
            }`}
          >
            <Save className={`w-4 h-4 ${isSaving ? 'animate-pulse' : ''}`} />
            {isSaving ? 'Saving...' : 'Save to Workspace'}
          </button>

          <button
            onClick={onVerifyCompile}
            disabled={isCompiling}
            className={`w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all border ${
              isCompiling 
                ? 'bg-slate-800 text-slate-500 border-slate-800 cursor-not-allowed' 
                : 'bg-slate-950/40 hover:bg-slate-800 border-slate-800 text-amber-400 hover:text-amber-300'
            }`}
          >
            <CheckCircle className={`w-4 h-4 ${isCompiling ? 'animate-spin' : ''}`} />
            {isCompiling ? 'Verifying...' : 'Verify & Compile Python'}
          </button>
        </div>
        
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => onExport('flat')}
            className="flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg border border-slate-800 bg-slate-950/40 hover:bg-slate-800 text-xs font-medium text-slate-200 transition-all"
            title="Export flat JSON array format"
          >
            <Download className="w-4 h-4 text-sky-400" />
            Export Flat
          </button>
          <button
            onClick={() => onExport('hierarchical')}
            className="flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg border border-slate-800 bg-slate-950/40 hover:bg-slate-800 text-xs font-medium text-slate-200 transition-all"
            title="Export hierarchical nested JSON format"
          >
            <Download className="w-4 h-4 text-purple-400" />
            Export Tree
          </button>
        </div>

        <input
          type="file"
          accept=".json"
          ref={fileInputRef}
          onChange={onImport}
          className="hidden"
        />

        <button
          onClick={triggerImportClick}
          className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg border border-slate-800 bg-slate-950/40 hover:bg-slate-800 text-xs font-medium text-slate-200 transition-all"
        >
          <Upload className="w-4 h-4 text-emerald-400" />
          Import JSON Architecture
        </button>

        <button
          onClick={onClear}
          className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg border border-red-950/50 bg-red-950/10 hover:bg-red-950/30 text-xs font-semibold text-red-400 transition-all"
        >
          <Trash2 className="w-4 h-4 text-red-500" />
          Clear Canvas
        </button>
      </div>
    </aside>
  );
};
