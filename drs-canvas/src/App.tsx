import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  ReactFlowProvider
} from '@xyflow/react';
import type { Connection, Edge, Node } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

// Import components
import { Sidebar } from './components/Sidebar';
import { Inspector } from './components/Inspector';

// Import custom nodes
import { ExtractionNode } from './nodes/ExtractionNode';
import { BufferNode } from './nodes/BufferNode';
import { FactoryNode } from './nodes/FactoryNode';
import { DataSourceNode } from './nodes/DataSourceNode';

const nodeTypes = {
  extractionNode: ExtractionNode,
  bufferNode: BufferNode,
  factoryNode: FactoryNode,
  dataSourceNode: DataSourceNode,
};

// Default initial nodes: Concentrator model from blending_modes_simulation.py
const initialNodes: Node[] = [
  {
    id: 'mine',
    type: 'extractionNode',
    position: { x: 80, y: 220 },
    data: {
      label: 'Concentrator Mine Face',
      class: 'ConcentratorMineFace',
      variables: {
        active_parcel_initial_mass: { class: 'Variable', value: 34975.28 },
        active_parcel_ore_fraction: { class: 'Variable', value: 0.7 },
        cumulative_extracted_mass: { class: 'Level', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        parcel_extracted_mass: { class: 'Level', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 }
      }
    }
  },
  {
    id: 'fleet',
    type: 'factoryNode',
    position: { x: 340, y: 200 },
    data: {
      label: 'Continuous Fleet Logistics',
      class: 'ContinuousFleetLogistics',
      variables: {
        stockpile2_routing_fraction: { class: 'Variable', value: 0.0 }
      }
    }
  },
  {
    id: 'ore1_stock',
    type: 'bufferNode',
    position: { x: 620, y: 100 },
    data: {
      label: 'Ore 1 Stockpile',
      class: 'Stockpile',
      variables: {
        current_mass: { class: 'Level', value: 42000.0, lower_threshold: 0.0, upper_threshold: 60000.0, rate: 0.0 },
        contained_ore_fraction_mass: { class: 'Level', value: 12600.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        actual_outflow_rate: { class: 'Variable', value: 0.0 }
      }
    }
  },
  {
    id: 'ore2_stock',
    type: 'bufferNode',
    position: { x: 620, y: 320 },
    data: {
      label: 'Ore 2 Stockpile',
      class: 'Stockpile',
      variables: {
        current_mass: { class: 'Level', value: 18000.0, lower_threshold: 0.0, upper_threshold: 60000.0, rate: 0.0 },
        contained_ore_fraction_mass: { class: 'Level', value: 5400.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        actual_outflow_rate: { class: 'Variable', value: 0.0 }
      }
    }
  },
  {
    id: 'plant',
    type: 'factoryNode',
    position: { x: 920, y: 220 },
    data: {
      label: 'Concentrator Plant',
      class: 'ConcentratorPlant',
      variables: {
        cumulative_milled_mass: { class: 'Level', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 }
      }
    }
  },
  {
    id: 'controller',
    type: 'factoryNode',
    position: { x: 340, y: 440 },
    data: {
      label: 'Concentrator Controller',
      class: 'ConcentratorController',
      variables: {
        active_operating_mode: { class: 'Variable', value: 'OperatingMode.MODE_A' },
        total_system_ore_mass: { class: 'Level', value: 60000.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        current_campaign_duration: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 1.0 },
        current_contingency_duration: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 1.0 },
        cumulative_time_mode_a: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 1.0 },
        cumulative_time_mode_b: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 1.0 },
        cumulative_time_shutdown: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 1.0 },
        target_mine_mass_rate: { class: 'Variable', value: 0.0 },
        target_stock1_outflow_rate: { class: 'Variable', value: 0.0 },
        target_stock2_outflow_rate: { class: 'Variable', value: 0.0 }
      }
    }
  }
];

const initialEdges: Edge[] = [
  {
    id: 'e-flow-mine-fleet',
    source: 'mine',
    sourceHandle: 'flow-out',
    target: 'fleet',
    targetHandle: 'flow-in',
    style: { stroke: '#3b82f6', strokeWidth: 4 }
  },
  {
    id: 'e-flow-fleet-ore1',
    source: 'fleet',
    sourceHandle: 'flow-out',
    target: 'ore1_stock',
    targetHandle: 'flow-in',
    style: { stroke: '#3b82f6', strokeWidth: 4 }
  },
  {
    id: 'e-flow-fleet-ore2',
    source: 'fleet',
    sourceHandle: 'flow-out',
    target: 'ore2_stock',
    targetHandle: 'flow-in',
    style: { stroke: '#3b82f6', strokeWidth: 4 }
  },
  {
    id: 'e-flow-ore1-plant',
    source: 'ore1_stock',
    sourceHandle: 'flow-out',
    target: 'plant',
    targetHandle: 'flow-in',
    style: { stroke: '#3b82f6', strokeWidth: 4 }
  },
  {
    id: 'e-flow-ore2-plant',
    source: 'ore2_stock',
    sourceHandle: 'flow-out',
    target: 'plant',
    targetHandle: 'flow-in',
    style: { stroke: '#3b82f6', strokeWidth: 4 }
  },
  {
    id: 'e-read-ore1-controller',
    source: 'ore1_stock',
    sourceHandle: 'read-out',
    target: 'controller',
    targetHandle: 'read-in',
    style: { stroke: '#f59e0b', strokeWidth: 1.5 }
  },
  {
    id: 'e-read-ore2-controller',
    source: 'ore2_stock',
    sourceHandle: 'read-out',
    target: 'controller',
    targetHandle: 'read-in',
    style: { stroke: '#f59e0b', strokeWidth: 1.5 }
  }
];

const LOCAL_STORAGE_KEY = 'drs-canvas-topology-v2';

const CanvasEditor = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNode, setSelectedNode] = useState<{ id: string; data: any } | null>(null);
  
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);

  // Load from local storage or defaults
  useEffect(() => {
    const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setNodes(parsed.nodes || []);
        setEdges(parsed.edges || []);
        return;
      } catch (e) {
        console.error('Error restoring localStorage topology, resetting to defaults.', e);
      }
    }
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [setNodes, setEdges]);

  // Persist to local storage
  const saveToLocalStorage = useCallback((newNodes: Node[], newEdges: Edge[]) => {
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify({ nodes: newNodes, edges: newEdges }));
  }, []);

  const updateNodesAndPersist = useCallback((updater: any) => {
    setNodes((nds) => {
      const next = typeof updater === 'function' ? updater(nds) : updater;
      saveToLocalStorage(next, edges);
      return next;
    });
  }, [edges, setNodes, saveToLocalStorage]);

  const updateEdgesAndPersist = useCallback((updater: any) => {
    setEdges((eds) => {
      const next = typeof updater === 'function' ? updater(eds) : updater;
      saveToLocalStorage(nodes, next);
      return next;
    });
  }, [nodes, setEdges, saveToLocalStorage]);

  const onConnect = useCallback(
    (params: Connection) => {
      // Determine style based on source handle type
      let edgeStyle: React.CSSProperties = {};
      if (params.sourceHandle?.startsWith('flow')) {
        edgeStyle = { stroke: '#3b82f6', strokeWidth: 4 };
      } else if (params.sourceHandle?.startsWith('read')) {
        edgeStyle = { stroke: '#f59e0b', strokeWidth: 1.5 };
      } else if (params.sourceHandle?.startsWith('data')) {
        edgeStyle = { stroke: '#10b981', strokeWidth: 2, strokeDasharray: '5,5' };
      }

      const newEdge = {
        ...params,
        style: edgeStyle,
      };

      updateEdgesAndPersist((eds: Edge[]) => addEdge(newEdge, eds));
    },
    [updateEdgesAndPersist]
  );

  const isValidConnection = useCallback((connection: Edge | Connection) => {
    const { sourceHandle, targetHandle } = connection;
    if (!sourceHandle || !targetHandle) return false;

    // Strict handle capability wiring:
    if (sourceHandle.startsWith('flow') && targetHandle.startsWith('flow')) return true;
    if (sourceHandle.startsWith('read') && targetHandle.startsWith('read')) return true;
    if (sourceHandle.startsWith('data') && targetHandle.startsWith('data')) return true;

    return false;
  }, []);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      if (!reactFlowWrapper.current || !reactFlowInstance) return;

      const type = event.dataTransfer.getData('application/reactflow');

      if (!type) return;

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      // Default variables by type
      let label = 'New Node';
      let nodeClass = 'Module';
      let variables = {};

      if (type === 'extractionNode') {
        label = 'New Extraction Face';
        nodeClass = 'ExtractionFace';
        variables = {
          rate: { class: 'Variable', value: 10.0 }
        };
      } else if (type === 'bufferNode') {
        label = 'New Stockpile';
        nodeClass = 'Stockpile';
        variables = {
          current_mass: { class: 'Level', value: 0.0, lower_threshold: 0.0, upper_threshold: 500.0, rate: 0.0 }
        };
      } else if (type === 'factoryNode') {
        label = 'New Concentrator';
        nodeClass = 'ProcessingPlant';
        variables = {
          throughput: { class: 'Variable', value: 100.0 }
        };
      } else if (type === 'dataSourceNode') {
        label = 'New Source';
        nodeClass = 'DataSource';
        variables = {
          batch_size: { class: 'Variable', value: 20.0 }
        };
      }

      // Generate a unique path-compliant ID
      const sanitizedLabel = label.toLowerCase().replace(/\s+/g, '_');
      let id = sanitizedLabel;
      let counter = 1;
      while (nodes.some((n) => n.id === id)) {
        id = `${sanitizedLabel}_${counter}`;
        counter++;
      }

      const newNode: Node = {
        id,
        type,
        position,
        data: {
          label,
          class: nodeClass,
          variables,
          layout: position
        },
      };

      updateNodesAndPersist((nds: Node[]) => nds.concat(newNode));
    },
    [reactFlowInstance, nodes, updateNodesAndPersist]
  );

  const onNodeClick = useCallback((_: any, node: Node) => {
    setSelectedNode({ id: node.id, data: node.data as any });
  }, []);

  const onUpdateNode = useCallback((nodeId: string, updatedData: any) => {
    updateNodesAndPersist((nds: Node[]) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          // Keep layout sync'ed
          return {
            ...node,
            data: {
              ...updatedData,
              layout: node.position
            }
          };
        }
        return node;
      })
    );
    setSelectedNode({ id: nodeId, data: updatedData });
  }, [updateNodesAndPersist]);

  const onNodeDragStop = useCallback((_: any, node: Node) => {
    updateNodesAndPersist((nds: Node[]) =>
      nds.map((n) => {
        if (n.id === node.id) {
          return {
            ...n,
            position: node.position,
            data: {
              ...n.data,
              layout: node.position
            }
          };
        }
        return n;
      })
    );
  }, [updateNodesAndPersist]);

  const onClear = useCallback(() => {
    if (window.confirm('Are you sure you want to clear the entire canvas?')) {
      setNodes([]);
      setEdges([]);
      setSelectedNode(null);
      localStorage.removeItem(LOCAL_STORAGE_KEY);
    }
  }, [setNodes, setEdges]);

  // Export JSON Structure
  const onExport = useCallback((format: 'flat' | 'hierarchical') => {
    if (nodes.length === 0) {
      alert('Canvas is empty.');
      return;
    }

    let exportData: any;

    if (format === 'flat') {
      // Flat array containing node configs and connection outputs
      exportData = nodes.map((node) => {
        const layout = node.position;
        const variables = node.data.variables || {};
        
        // Find links
        const flow_inputs = edges
          .filter((e) => e.target === node.id && e.targetHandle === 'flow-in')
          .map((e) => e.source);
        
        const data_inputs = edges
          .filter((e) => e.target === node.id && e.targetHandle === 'data-in')
          .map((e) => e.source);

        const variable_reads = edges
          .filter((e) => e.target === node.id && e.targetHandle === 'read-in')
          .map((e) => {
            // Find variable name read. Default to a guess based on source variables
            const srcNode = nodes.find(n => n.id === e.source);
            const varNames = srcNode ? Object.keys(srcNode.data.variables || {}) : [];
            const readVar = varNames.length > 0 ? varNames[0] : 'value';
            return {
              module: e.source,
              variable: readVar
            };
          });

        return {
          id: node.id,
          class: node.data.class,
          layout,
          variables,
          connections: {
            flow_inputs,
            data_inputs,
            variable_reads
          }
        };
      });
    } else {
      // Hierarchical structure. Reconstruct tree structure from dot notation
      // First, build a map of paths to children
      const buildTree = () => {
        const root: any = { class: 'DRSModel', layout: { x: 0, y: 0 }, variables: {}, children: {}, connections: {} };
        
        // Populate nodes into tree
        nodes.forEach((node) => {
          const parts = node.id.split('.');
          let curr = root;
          
          parts.forEach((part: string, index: number) => {
            if (index === parts.length - 1) {
              // Leaf module definition
              const flow_inputs = edges
                .filter((e) => e.target === node.id && e.targetHandle === 'flow-in')
                .map((e) => e.source);
              const data_inputs = edges
                .filter((e) => e.target === node.id && e.targetHandle === 'data-in')
                .map((e) => e.source);
              const variable_reads = edges
                .filter((e) => e.target === node.id && e.targetHandle === 'read-in')
                .map((e) => {
                  const srcNode = nodes.find(n => n.id === e.source);
                  const varNames = srcNode ? Object.keys(srcNode.data.variables || {}) : [];
                  return {
                    module: e.source,
                    variable: varNames[0] || 'value'
                  };
                });

              curr.children[part] = {
                class: node.data.class,
                layout: node.position,
                variables: node.data.variables || {},
                children: {},
                connections: {
                  flow_inputs,
                  data_inputs,
                  variable_reads
                }
              };
            } else {
              // Ensure intermediate nodes exist
              if (!curr.children[part]) {
                curr.children[part] = {
                  class: 'Module',
                  layout: { x: 0, y: 0 },
                  variables: {},
                  children: {},
                  connections: {}
                };
              }
              curr = curr.children[part];
            }
          });
        });
        
        return root;
      };
      
      exportData = buildTree();
    }

    const jsonStr = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `drs_topology_${format}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [nodes, edges]);

  // Import JSON Structure
  const onImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        const importedNodes: Node[] = [];
        const importedEdges: Edge[] = [];

        // Check if flat array or tree
        if (Array.isArray(parsed)) {
          // Flat list
          parsed.forEach((item: any, index: number) => {
            const position = item.layout || { x: 100 + index * 150, y: 150 };
            
            // Map class name to custom visual type
            let type = 'extractionNode';
            const cls = item.class || '';
            if (cls.includes('Stock') || cls.includes('Buffer') || cls.includes('Crusher')) {
              type = 'bufferNode';
            } else if (cls.includes('Plant') || cls.includes('Factory') || cls.includes('Concentrator')) {
              type = 'factoryNode';
            } else if (cls.includes('Source') || cls.includes('Stream')) {
              type = 'dataSourceNode';
            }

            importedNodes.push({
              id: item.id || `node_${index}`,
              type,
              position,
              data: {
                label: item.id || `Node ${index}`,
                class: item.class || 'Module',
                variables: item.variables || {},
                layout: position
              }
            });

            // Reconstruct connections if present
            const conns = item.connections || {};
            (conns.flow_inputs || []).forEach((src: string, eIdx: number) => {
              importedEdges.push({
                id: `e-flow-${src}-${item.id}-${eIdx}`,
                source: src,
                sourceHandle: 'flow-out',
                target: item.id,
                targetHandle: 'flow-in',
                style: { stroke: '#3b82f6', strokeWidth: 4 }
              });
            });
            (conns.data_inputs || []).forEach((src: string, eIdx: number) => {
              importedEdges.push({
                id: `e-data-${src}-${item.id}-${eIdx}`,
                source: src,
                sourceHandle: 'data-out',
                target: item.id,
                targetHandle: 'data-in',
                style: { stroke: '#10b981', strokeWidth: 2, strokeDasharray: '5,5' }
              });
            });
            (conns.variable_reads || []).forEach((srcObj: { module: string }, eIdx: number) => {
              importedEdges.push({
                id: `e-read-${srcObj.module}-${item.id}-${eIdx}`,
                source: srcObj.module,
                sourceHandle: 'read-out',
                target: item.id,
                targetHandle: 'read-in',
                style: { stroke: '#f59e0b', strokeWidth: 1.5 }
              });
            });
          });
        } else {
          // Hierarchical tree reconstruction
          // Walk tree recursively to flatten
          let nodeIndex = 0;
          const flattenTree = (node: any, path: string = '') => {
            const currentPath = path ? path : '';
            
            if (node.class && node.class !== 'DRSModel') {
              const position = node.layout || { x: 100 + (nodeIndex % 4) * 250, y: 100 + Math.floor(nodeIndex / 4) * 180 };
              
              let type = 'extractionNode';
              const cls = node.class || '';
              if (cls.includes('Stock') || cls.includes('Buffer') || cls.includes('Crusher')) {
                type = 'bufferNode';
              } else if (cls.includes('Plant') || cls.includes('Factory') || cls.includes('Concentrator')) {
                type = 'factoryNode';
              } else if (cls.includes('Source') || cls.includes('Stream')) {
                type = 'dataSourceNode';
              }

              importedNodes.push({
                id: currentPath,
                type,
                position,
                data: {
                  label: currentPath.split('.').pop() || 'Node',
                  class: node.class || 'Module',
                  variables: node.variables || {},
                  layout: position
                }
              });

              // Connections
              const conns = node.connections || {};
              (conns.flow_inputs || []).forEach((src: string, eIdx: number) => {
                importedEdges.push({
                  id: `e-flow-${src}-${currentPath}-${eIdx}`,
                  source: src,
                  sourceHandle: 'flow-out',
                  target: currentPath,
                  targetHandle: 'flow-in',
                  style: { stroke: '#3b82f6', strokeWidth: 4 }
                });
              });
              (conns.data_inputs || []).forEach((src: string, eIdx: number) => {
                importedEdges.push({
                  id: `e-data-${src}-${currentPath}-${eIdx}`,
                  source: src,
                  sourceHandle: 'data-out',
                  target: currentPath,
                  targetHandle: 'data-in',
                  style: { stroke: '#10b981', strokeWidth: 2, strokeDasharray: '5,5' }
                });
              });
              (conns.variable_reads || []).forEach((srcObj: { module: string }, eIdx: number) => {
                importedEdges.push({
                  id: `e-read-${srcObj.module}-${currentPath}-${eIdx}`,
                  source: srcObj.module,
                  sourceHandle: 'read-out',
                  target: currentPath,
                  targetHandle: 'read-in',
                  style: { stroke: '#f59e0b', strokeWidth: 1.5 }
                });
              });

              nodeIndex++;
            }

            // Children traversal
            if (node.children) {
              Object.entries(node.children).forEach(([name, childNode]) => {
                const childPath = currentPath ? `${currentPath}.${name}` : name;
                flattenTree(childNode, childPath);
              });
            }
          };

          flattenTree(parsed);
        }

        setNodes(importedNodes);
        setEdges(importedEdges);
        saveToLocalStorage(importedNodes, importedEdges);
        setSelectedNode(null);
      } catch (err) {
        alert(`Error importing JSON: ${err}`);
      }
    };
    reader.readAsText(file);
    e.target.value = ''; // clear input
  }, [setNodes, setEdges, saveToLocalStorage]);

  return (
    <div className="flex w-screen h-screen bg-slate-950 font-sans text-slate-100 overflow-hidden">
      {/* Sidebar palette & controls */}
      <Sidebar onExport={onExport} onImport={onImport} onClear={onClear} />

      {/* React Flow Editor Grid */}
      <div className="flex-1 h-full relative" ref={reactFlowWrapper}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onPaneClick={() => setSelectedNode(null)}
          onNodeDragStop={onNodeDragStop}
          onInit={setReactFlowInstance}
          onDrop={onDrop}
          onDragOver={onDragOver}
          nodeTypes={nodeTypes}
          isValidConnection={isValidConnection}
          fitView
          colorMode="dark"
        >
          <Controls className="!bg-slate-900 !border-slate-800 !text-white [&_button]:!border-slate-800 [&_button]:!bg-slate-900 [&_button:hover]:!bg-slate-800" />
          <MiniMap 
            nodeColor={(node) => {
              if (node.type === 'extractionNode') return '#f59e0b';
              if (node.type === 'bufferNode') return '#0ea5e9';
              if (node.type === 'factoryNode') return '#c084fc';
              if (node.type === 'dataSourceNode') return '#10b981';
              return '#64748b';
            }}
            maskColor="rgba(15, 23, 42, 0.7)"
            className="!bg-slate-900 !border-slate-800 rounded-lg shadow-xl"
          />
          <Background color="#334155" gap={20} size={1} />
        </ReactFlow>
      </div>

      {/* Editor Inspector Panel */}
      {selectedNode && (
        <Inspector
          selectedNode={selectedNode as any}
          onUpdateNode={onUpdateNode}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </div>
  );
};

export default () => (
  <ReactFlowProvider>
    <CanvasEditor />
  </ReactFlowProvider>
);
