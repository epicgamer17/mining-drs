import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
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
import { Play, Pause, RotateCcw, ListFilter } from 'lucide-react';

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
      attributes: {
        name: 'Ore1Stock',
        expected_attributes: ['contained_ore_fraction_mass'],
      },
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
      attributes: {
        name: 'Ore2Stock',
        expected_attributes: ['contained_ore_fraction_mass'],
      },
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
        active_operating_mode: { class: 'Variable', value: { __type__: 'OperatingMode', name: 'MODE_A' } },
        total_system_ore_mass: { class: 'Level', value: 60000.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        current_campaign_duration: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        current_contingency_duration: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        cumulative_time_mode_a: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        cumulative_time_mode_a_contingency: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        cumulative_time_mode_a_surging: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        cumulative_time_mode_b: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        cumulative_time_mode_b_contingency: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        cumulative_time_mode_b_surging: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
        cumulative_time_shutdown: { class: 'Timer', value: 0.0, lower_threshold: '-Infinity', upper_threshold: 'Infinity', rate: 0.0 },
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
    data: { param: 'inflow' },
    style: { stroke: '#3b82f6', strokeWidth: 4 }
  },
  {
    id: 'e-flow-fleet-ore2',
    source: 'fleet',
    sourceHandle: 'flow-out',
    target: 'ore2_stock',
    targetHandle: 'flow-in',
    data: { param: 'inflow' },
    style: { stroke: '#3b82f6', strokeWidth: 4 }
  },
  {
    id: 'e-flow-ore1-plant',
    source: 'ore1_stock',
    sourceHandle: 'flow-out',
    target: 'plant',
    targetHandle: 'flow-in',
    data: { param: 'ore1_outflow' },
    style: { stroke: '#3b82f6', strokeWidth: 4 }
  },
  {
    id: 'e-flow-ore2-plant',
    source: 'ore2_stock',
    sourceHandle: 'flow-out',
    target: 'plant',
    targetHandle: 'flow-in',
    data: { param: 'ore2_outflow' },
    style: { stroke: '#3b82f6', strokeWidth: 4 }
  },
  {
    id: 'e-read-ore1-controller',
    source: 'ore1_stock',
    sourceHandle: 'read-out',
    target: 'controller',
    targetHandle: 'read-in',
    data: { variable: 'current_mass' },
    style: { stroke: '#f59e0b', strokeWidth: 1.5 }
  },
  {
    id: 'e-read-ore2-controller',
    source: 'ore2_stock',
    sourceHandle: 'read-out',
    target: 'controller',
    targetHandle: 'read-in',
    data: { variable: 'current_mass' },
    style: { stroke: '#f59e0b', strokeWidth: 1.5 }
  },
  {
    id: 'e-data-controller-ore1',
    source: 'controller',
    sourceHandle: 'data-out',
    target: 'ore1_stock',
    targetHandle: 'data-in',
    data: { param: 'requested_outflow_rate', variable: 'target_stock1_outflow_rate' },
    style: { stroke: '#10b981', strokeWidth: 2, strokeDasharray: '5,5' }
  },
  {
    id: 'e-data-controller-ore2',
    source: 'controller',
    sourceHandle: 'data-out',
    target: 'ore2_stock',
    targetHandle: 'data-in',
    data: { param: 'requested_outflow_rate', variable: 'target_stock2_outflow_rate' },
    style: { stroke: '#10b981', strokeWidth: 2, strokeDasharray: '5,5' }
  }
];

const LOCAL_STORAGE_KEY = 'drs-canvas-topology-v2';
const SCHEMA_VERSION = 2;

const CanvasEditor = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNode, setSelectedNode] = useState<{ id: string; data: any } | null>(null);
  const [isCompiling, setIsCompiling] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [currentParentId, setCurrentParentId] = useState<string | null>(null);

  // Telemetry playback states
  const [telemetryHistory, setTelemetryHistory] = useState<any[]>([]);
  const [telemetryEvents, setTelemetryEvents] = useState<any[]>([]);
  const [currentPlaybackTime, setCurrentPlaybackTime] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [maxSimTime, setMaxSimTime] = useState<number>(99999);
  const [simSeed, setSimSeed] = useState<number>(42);
  const [showEventLog, setShowEventLog] = useState<boolean>(false);
  const [dashboardPng, setDashboardPng] = useState<string | null>(null);
  const [showDashboard, setShowDashboard] = useState<boolean>(false);
  
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);

  // Persist to local storage
  const saveToLocalStorage = useCallback((newNodes: Node[], newEdges: Edge[]) => {
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify({ version: SCHEMA_VERSION, nodes: newNodes, edges: newEdges }));
  }, []);

  const migrateReadEdgeMetadata = useCallback((nextNodes: Node[], nextEdges: Edge[]) => {
    return nextEdges.map((edge) => {
      if (edge.targetHandle !== 'read-in' || (edge.data as any)?.variable) {
        return edge;
      }

      const sourceNode = nextNodes.find((node) => node.id === edge.source);
      const sourceVariables = (sourceNode?.data as any)?.variables || {};
      const variableNames = Object.keys(sourceVariables);
      let variable: string | undefined;

      if ('current_mass' in sourceVariables) {
        variable = 'current_mass';
      } else if (variableNames.length === 1) {
        variable = variableNames[0];
      }

      if (!variable) {
        return edge;
      }

      return {
        ...edge,
        data: {
          ...(edge.data || {}),
          variable,
        },
      };
    });
  }, []);

  // Load from dev server with fallback to localStorage or defaults
  useEffect(() => {
    const loadTopology = async () => {
      try {
        const res = await fetch('/api/topology');
        if (res.ok) {
          const parsed = await res.json();
          if (parsed && Array.isArray(parsed.nodes)) {
            const migratedEdges = migrateReadEdgeMetadata(parsed.nodes, parsed.edges || []);
            setNodes(parsed.nodes);
            setEdges(migratedEdges);
            saveToLocalStorage(parsed.nodes, migratedEdges);
            return;
          }
        }
      } catch (err) {
        console.warn('Could not load topology from dev server, falling back to localStorage.', err);
      }

      const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          if (parsed.version !== SCHEMA_VERSION) {
            console.warn(`Schema version mismatch (cached: ${parsed.version}, expected: ${SCHEMA_VERSION}), resetting to defaults.`);
            localStorage.removeItem(LOCAL_STORAGE_KEY);
          } else {
            const savedNodes = parsed.nodes || [];
            const migratedEdges = migrateReadEdgeMetadata(savedNodes, parsed.edges || []);
            setNodes(savedNodes);
            setEdges(migratedEdges);
            saveToLocalStorage(savedNodes, migratedEdges);
            return;
          }
        } catch (e) {
          console.error('Error restoring localStorage topology, resetting to defaults.', e);
        }
      }
      setNodes(initialNodes);
      setEdges(initialEdges);
    };

    loadTopology();
  }, [setNodes, setEdges, migrateReadEdgeMetadata, saveToLocalStorage]);

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

  const onSaveToWorkspace = useCallback(async () => {
    setIsSaving(true);
    try {
      const res = await fetch('/api/topology', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ nodes, edges })
      });
      const data = await res.json();
      if (res.ok && data.status === 'ok') {
        alert('Successfully saved canvas and translated topology to workspace!');
      } else {
        alert(`Failed to save: ${data.error || 'Unknown error'}`);
      }
    } catch (err) {
      alert(`Network error saving to workspace: ${err}`);
    } finally {
      setIsSaving(false);
    }
  }, [nodes, edges]);

  const onVerifyCompile = useCallback(async () => {
    setIsCompiling(true);
    try {
      const res = await fetch('/api/compile', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ nodes, edges })
      });
      const data = await res.json();
      if (res.ok && data.status === 'ok') {
        alert(`Verification Success!\n\n${data.message}`);
      } else {
        const details = data.details ? `\n\nDetails:\n${data.details}` : '';
        alert(`Verification Failed!\n\nError: ${data.message}${details}`);
      }
    } catch (err) {
      alert(`Network error verifying compilation: ${err}`);
    } finally {
      setIsCompiling(false);
    }
  }, [nodes, edges]);

  const onRunSimulation = useCallback(async () => {
    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ nodes, edges, max_time: maxSimTime, seed: simSeed }),
      });
      const data = await res.json();
      if (res.ok && data.status === 'ok') {
        setTelemetryHistory(data.history || []);
        setTelemetryEvents(data.events || []);
        const rawPng = data.plots?.dashboard_png;
        setDashboardPng(rawPng ? `data:image/png;base64,${rawPng}` : null);
        setCurrentPlaybackTime(0);
        setIsPlaying(true);
        alert('Simulation execution complete! Telemetry loaded.');
      } else {
        alert(`Simulation execution failed: ${data.message || 'Unknown error'}`);
      }
    } catch (err) {
      alert(`Network error running simulation: ${err}`);
    }
  }, [nodes, edges, maxSimTime, simSeed]);

  // Playback timer loop effect
  useEffect(() => {
    if (!isPlaying || telemetryHistory.length === 0) return;

    const intervalTime = 100; // 100ms ticks
    const interval = setInterval(() => {
      setCurrentPlaybackTime((t) => {
        const maxTime = telemetryHistory[telemetryHistory.length - 1]?.time || maxSimTime;
        const nextTime = t + (intervalTime / 1000) * playbackSpeed;
        if (nextTime >= maxTime) {
          setIsPlaying(false);
          return maxTime;
        }
        return nextTime;
      });
    }, intervalTime);

    return () => clearInterval(interval);
  }, [isPlaying, telemetryHistory, playbackSpeed, maxSimTime]);

  // Find current active telemetry frame
  const currentFrame = useMemo(() => {
    if (telemetryHistory.length === 0) return null;
    let closest = telemetryHistory[0];
    let minDiff = Math.abs(closest.time - currentPlaybackTime);
    for (let i = 1; i < telemetryHistory.length; i++) {
      const frame = telemetryHistory[i];
      const diff = Math.abs(frame.time - currentPlaybackTime);
      if (diff < minDiff) {
        closest = frame;
        minDiff = diff;
      }
    }
    return closest;
  }, [telemetryHistory, currentPlaybackTime]);

  const onConnect = useCallback(
    (params: Connection) => {
      // Determine style based on source handle type
      let edgeStyle: React.CSSProperties = {};
      let edgeData: Record<string, unknown> | undefined;
      if (params.sourceHandle?.startsWith('flow')) {
        edgeStyle = { stroke: '#3b82f6', strokeWidth: 4 };
        const param = window.prompt('Parameter name on target (e.g., inflow, ore1_outflow):');
        if (param) edgeData = { param };
      } else if (params.sourceHandle?.startsWith('read')) {
        edgeStyle = { stroke: '#f59e0b', strokeWidth: 1.5 };
        const sourceNode = nodes.find((node) => node.id === params.source);
        const variableNames = Object.keys((sourceNode?.data as any)?.variables || {});
        if (variableNames.length === 0) {
          alert('Read edges require the source node to expose at least one variable.');
          return;
        }
        let readVariable = variableNames[0];
        if (variableNames.length > 1) {
          const selected = window.prompt(
            `Variable to read from ${params.source}:\n${variableNames.join(', ')}`,
            variableNames[0]
          );
          if (!selected) return;
          if (!variableNames.includes(selected)) {
            alert(`"${selected}" is not a variable on ${params.source}.`);
            return;
          }
          readVariable = selected;
        }
        edgeData = { variable: readVariable };
      } else if (params.sourceHandle?.startsWith('data')) {
        edgeStyle = { stroke: '#10b981', strokeWidth: 2, strokeDasharray: '5,5' };
        const param = window.prompt('Parameter name on target (e.g., requested_outflow_rate):');
        if (!param) return;
        const variable = window.prompt('Source variable name (e.g., target_stock1_outflow_rate):');
        if (!variable) return;
        edgeData = { param, variable };
      }

      const newEdge = {
        ...params,
        data: edgeData,
        style: edgeStyle,
      };

      updateEdgesAndPersist((eds: Edge[]) => addEdge(newEdge, eds));
    },
    [nodes, updateEdgesAndPersist]
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

      // Default variables and attributes by type
      let label = 'New Node';
      let nodeClass = 'Module';
      let variables = {};
      let attributes = {};

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
        attributes = {
          name: label.toLowerCase().replace(/\s+/g, '_'),
          expected_attributes: [],
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
      let relativeId = sanitizedLabel;
      let counter = 1;
      
      const makeFullId = (relId: string) => currentParentId ? `${currentParentId}.${relId}` : relId;

      while (nodes.some((n) => n.id === makeFullId(relativeId))) {
        relativeId = `${sanitizedLabel}_${counter}`;
        counter++;
      }
      
      const id = makeFullId(relativeId);

      const newNode: Node = {
        id,
        type,
        position,
        data: {
          label,
          class: nodeClass,
          variables,
          attributes,
          layout: position
        },
      };

      updateNodesAndPersist((nds: Node[]) => nds.concat(newNode));
    },
    [reactFlowInstance, nodes, updateNodesAndPersist, currentParentId]
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
    if (node.id.startsWith('proxy-')) return;
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
      // Track output_index per source for flow-in connections (tuple unpacking)
      const sourceConsumerCount: Record<string, number> = {};
      
      exportData = nodes.map((node) => {
        const layout = node.position;
        const variables = node.data.variables || {};
        
        // Find links with dict format (module, param, output_index/variable)
        const flow_inputs = edges
          .filter((e) => e.target === node.id && e.targetHandle === 'flow-in')
          .map((e) => {
            const src = e.source;
            if (sourceConsumerCount[src] === undefined) sourceConsumerCount[src] = 0;
            const outputIndex = sourceConsumerCount[src];
            sourceConsumerCount[src]++;
            return {
              module: src,
              param: (e.data as any)?.param ?? null,
              output_index: outputIndex
            };
          });
        
        const data_inputs = edges
          .filter((e) => e.target === node.id && e.targetHandle === 'data-in')
          .map((e) => ({
            module: e.source,
            param: (e.data as any)?.param ?? null,
            variable: (e.data as any)?.variable ?? null,
          }));

        const variable_reads = edges
          .filter((e) => e.target === node.id && e.targetHandle === 'read-in')
          .map((e) => {
            const readVar = (e.data as any)?.variable;
            if (!readVar) {
              throw new Error(`Read edge ${e.id} is missing data.variable.`);
            }
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
          attributes: (node.data as any).attributes || {},
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
        const sourceConsumerCount: Record<string, number> = {};
        
        // Populate nodes into tree
        nodes.forEach((node) => {
          const parts = node.id.split('.');
          let curr = root;
          
          parts.forEach((part: string, index: number) => {
            if (index === parts.length - 1) {
              // Leaf module definition
              const flow_inputs = edges
                .filter((e) => e.target === node.id && e.targetHandle === 'flow-in')
                .map((e) => {
                  const src = e.source;
                  if (sourceConsumerCount[src] === undefined) sourceConsumerCount[src] = 0;
                  const outputIndex = sourceConsumerCount[src];
                  sourceConsumerCount[src]++;
                  return {
                    module: src,
                    param: (e.data as any)?.param ?? null,
                    output_index: outputIndex
                  };
                });
              const data_inputs = edges
                .filter((e) => e.target === node.id && e.targetHandle === 'data-in')
                .map((e) => ({
                  module: e.source,
                  param: (e.data as any)?.param ?? null,
                  variable: (e.data as any)?.variable ?? null,
                }));
              const variable_reads = edges
                .filter((e) => e.target === node.id && e.targetHandle === 'read-in')
                .map((e) => {
                  const readVar = (e.data as any)?.variable;
                  if (!readVar) {
                    throw new Error(`Read edge ${e.id} is missing data.variable.`);
                  }
                  return {
                    module: e.source,
                    variable: readVar
                  };
                });

              curr.children[part] = {
                class: node.data.class,
                layout: node.position,
                variables: node.data.variables || {},
                attributes: (node.data as any).attributes || {},
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

        // Helper to check layout validity and do automatic column alignments
        const getLayoutPosition = (layout: any, cls: string, idx: number) => {
          if (layout && typeof layout.x === 'number' && typeof layout.y === 'number') {
            return layout;
          }
          let col = 1;
          if (cls.includes('Source') || cls.includes('Generator') || cls.includes('Scheduler')) {
            col = 0;
          } else if (cls.includes('Face') || cls.includes('Extractor') || cls.includes('Mine')) {
            col = 1;
          } else if (cls.includes('Stock') || cls.includes('Buffer') || cls.includes('Crusher') || cls.includes('Logistics')) {
            col = 2;
          } else if (cls.includes('Plant') || cls.includes('Factory') || cls.includes('Concentrator') || cls.includes('Controller')) {
            col = 3;
          }
          return { x: 100 + col * 280, y: 100 + (idx * 160) % 500 };
        };

        // Check if flat array or tree
        if (Array.isArray(parsed)) {
          // Flat list
          parsed.forEach((item: any, index: number) => {
            const cls = item.class || '';
            const position = getLayoutPosition(item.layout, cls, index);
            
            // Map class name to custom visual type
            let type = 'extractionNode';
            if (cls.includes('Stock') || cls.includes('Buffer') || cls.includes('Crusher')) {
              type = 'bufferNode';
            } else if (cls.includes('Plant') || cls.includes('Factory') || cls.includes('Concentrator') || cls.includes('Controller')) {
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
                attributes: item.attributes || {},
                layout: position
              }
            });

            // Reconstruct connections if present
            // Support both old (string) and new (dict) formats
            const conns = item.connections || {};
            (conns.flow_inputs || []).forEach((entry: any, eIdx: number) => {
              const src = typeof entry === 'string' ? entry : entry.module;
              importedEdges.push({
                id: `e-flow-${src}-${item.id}-${eIdx}`,
                source: src,
                sourceHandle: 'flow-out',
                target: item.id,
                targetHandle: 'flow-in',
                style: { stroke: '#3b82f6', strokeWidth: 4 }
              });
            });
            (conns.data_inputs || []).forEach((entry: any, eIdx: number) => {
              const src = typeof entry === 'string' ? entry : entry.module;
              const edgeData: Record<string, unknown> = {};
              if (typeof entry === 'object' && entry !== null) {
                if (entry.param) edgeData.param = entry.param;
                if (entry.variable) edgeData.variable = entry.variable;
              }
              importedEdges.push({
                id: `e-data-${src}-${item.id}-${eIdx}`,
                source: src,
                sourceHandle: 'data-out',
                target: item.id,
                targetHandle: 'data-in',
                data: Object.keys(edgeData).length > 0 ? edgeData : undefined,
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
                data: { variable: (srcObj as any).variable },
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
              const cls = node.class || '';
              const position = getLayoutPosition(node.layout, cls, nodeIndex);
              
              let type = 'extractionNode';
              if (cls.includes('Stock') || cls.includes('Buffer') || cls.includes('Crusher')) {
                type = 'bufferNode';
              } else if (cls.includes('Plant') || cls.includes('Factory') || cls.includes('Concentrator') || cls.includes('Controller')) {
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
                  attributes: node.attributes || {},
                  layout: position
                }
              });

              // Connections
              // Support both old (string) and new (dict) formats
              const conns = node.connections || {};
              (conns.flow_inputs || []).forEach((entry: any, eIdx: number) => {
                const src = typeof entry === 'string' ? entry : entry.module;
                importedEdges.push({
                  id: `e-flow-${src}-${currentPath}-${eIdx}`,
                  source: src,
                  sourceHandle: 'flow-out',
                  target: currentPath,
                  targetHandle: 'flow-in',
                  style: { stroke: '#3b82f6', strokeWidth: 4 }
                });
              });
              (conns.data_inputs || []).forEach((entry: any, eIdx: number) => {
                const src = typeof entry === 'string' ? entry : entry.module;
                const edgeData: Record<string, unknown> = {};
                if (typeof entry === 'object' && entry !== null) {
                  if (entry.param) edgeData.param = entry.param;
                  if (entry.variable) edgeData.variable = entry.variable;
                }
                importedEdges.push({
                  id: `e-data-${src}-${currentPath}-${eIdx}`,
                  source: src,
                  sourceHandle: 'data-out',
                  target: currentPath,
                  targetHandle: 'data-in',
                  data: Object.keys(edgeData).length > 0 ? edgeData : undefined,
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
                  data: { variable: (srcObj as any).variable },
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

  // 1. Identify which nodes are containers (contain children with dot-prefixed IDs)
  const containerNodeIds = new Set<string>();
  nodes.forEach(n => {
    if (n.id.includes('.')) {
      const parts = n.id.split('.');
      for (let i = 1; i < parts.length; i++) {
        containerNodeIds.add(parts.slice(0, i).join('.'));
      }
    }
  });

  const getEdgeFlowRate = (src: string) => {
    if (!currentFrame) return 0;
    const possibleKeys = [
      `${src}.rate`,
      `${src}.throughput`,
      `${src}.actual_outflow_rate`,
      `${src}.outflow_rate`,
      `${src}.value`
    ];
    for (const key of possibleKeys) {
      if (currentFrame[key] !== undefined) {
        return currentFrame[key];
      }
    }
    const fallbackKey = Object.keys(currentFrame).find(k => k.startsWith(src + '.') && (k.endsWith('rate') || k.endsWith('throughput')));
    if (fallbackKey && currentFrame[fallbackKey] !== undefined) {
      return currentFrame[fallbackKey];
    }
    return 0;
  };

  const getEdgeStyle = (src: string, originalStyle: React.CSSProperties = {}) => {
    if (!currentFrame) return originalStyle;
    const flowRate = getEdgeFlowRate(src);
    const isFlowActive = flowRate > 0;
    
    if (isFlowActive) {
      return {
        ...originalStyle,
        stroke: '#3b82f6',
        strokeWidth: Math.min(8, 2.5 + flowRate / 15),
        strokeDasharray: '6,6',
        animation: `dash ${Math.max(0.2, 5 / flowRate)}s linear infinite`
      };
    } else {
      return {
        ...originalStyle,
        opacity: telemetryHistory.length > 0 ? 0.4 : 1.0
      };
    }
  };

  // Filter nodes belonging to the current parent level
  const baseVisibleNodes = nodes.filter(node => {
    if (currentParentId === null) {
      return !node.id.includes('.');
    } else {
      const prefix = currentParentId + '.';
      if (node.id.startsWith(prefix)) {
        const rest = node.id.substring(prefix.length);
        return !rest.includes('.');
      }
      return false;
    }
  }).map(node => {
    const isContainer = containerNodeIds.has(node.id) || ['ConcentratorPlant', 'DRSModel', 'Module', 'ProcessingPlant', 'ContinuousFleetLogistics', 'ConcentratorController', 'ConcentratorModel'].includes((node.data as any).class);
    
    // Inject telemetry values into node data variables
    let nodeVariables = { ...(node.data as any).variables };
    if (currentFrame) {
      Object.keys(nodeVariables).forEach(varName => {
        const telemetryKey = `${node.id}.${varName}`;
        if (currentFrame[telemetryKey] !== undefined) {
          let rawVal = currentFrame[telemetryKey];
          // Handle string modes or numbers
          let displayVal = typeof rawVal === 'number' ? Number(rawVal.toFixed(2)) : rawVal;
          nodeVariables[varName] = {
            ...nodeVariables[varName],
            value: displayVal
          };
        }
      });
    }

    return {
      ...node,
      data: {
        ...node.data,
        isContainer,
        variables: nodeVariables
      }
    };
  });

  // Construct final visibleNodes and visibleEdges by resolving boundary proxies
  const visibleNodes: Node[] = [...baseVisibleNodes];
  const visibleEdges: Edge[] = [];

  const proxyInputs: { [sourceId: string]: string } = {}; // sourceId -> proxyNodeId
  const proxyOutputs: { [targetId: string]: string } = {}; // targetId -> proxyNodeId

  edges.forEach(edge => {
    const source = edge.source;
    const target = edge.target;

    if (currentParentId === null) {
      // Root level view: if edge connects two nodes, check if they are nested submodules.
      // If so, re-route the edge to their parent container at the root level!
      const rootSource = source.includes('.') ? source.split('.')[0] : source;
      const rootTarget = target.includes('.') ? target.split('.')[0] : target;

      if (rootSource !== rootTarget) {
        const rootEdgeId = `e-h-${rootSource}-${rootTarget}`;
        // Prevent duplicate edges at parent level
        if (!visibleEdges.some(e => e.id === rootEdgeId)) {
          visibleEdges.push({
            id: rootEdgeId,
            source: rootSource,
            sourceHandle: edge.sourceHandle?.startsWith('flow') ? 'flow-out' : edge.sourceHandle?.startsWith('read') ? 'read-out' : 'data-out',
            target: rootTarget,
            targetHandle: edge.targetHandle?.startsWith('flow') ? 'flow-in' : edge.targetHandle?.startsWith('read') ? 'read-in' : 'data-in',
            style: getEdgeStyle(source, edge.style),
          });
        }
      } else {
        // Purely internal edge between siblings at root level
        if (!source.includes('.') && !target.includes('.')) {
          visibleEdges.push({
            ...edge,
            style: getEdgeStyle(source, edge.style)
          });
        }
      }
    } else {
      // Drilled down inside currentParentId
      const isSourceInSub = source.startsWith(currentParentId + '.') && !source.substring(currentParentId.length + 1).includes('.');
      const isTargetInSub = target.startsWith(currentParentId + '.') && !target.substring(currentParentId.length + 1).includes('.');

      if (isSourceInSub && isTargetInSub) {
        // Edge is fully inside this sub-circuit
        visibleEdges.push({
          ...edge,
          style: getEdgeStyle(source, edge.style)
        });
      } else if (isTargetInSub && !isSourceInSub) {
        // Incoming edge from outside this sub-circuit
        const proxyId = `proxy-in-${source}`;
        if (!proxyInputs[source]) {
          proxyInputs[source] = proxyId;
          const label = source.split('.').pop() || source;
          visibleNodes.push({
            id: proxyId,
            type: 'dataSourceNode',
            position: { x: 20, y: 100 + Object.keys(proxyInputs).length * 150 },
            data: {
              label: `From ${label}`,
              class: 'ProxyInput',
              variables: {}
            },
            style: { borderStyle: 'dashed', opacity: 0.8 }
          });
        }
        visibleEdges.push({
          ...edge,
          id: `e-proxy-in-${source}-${target}`,
          source: proxyId,
          sourceHandle: 'flow-out',
          style: getEdgeStyle(source, edge.style)
        });
      } else if (isSourceInSub && !isTargetInSub) {
        // Outgoing edge to outside this sub-circuit
        const proxyId = `proxy-out-${target}`;
        if (!proxyOutputs[target]) {
          proxyOutputs[target] = proxyId;
          const label = target.split('.').pop() || target;
          visibleNodes.push({
            id: proxyId,
            type: 'factoryNode',
            position: { x: 950, y: 100 + Object.keys(proxyOutputs).length * 150 },
            data: {
              label: `To ${label}`,
              class: 'ProxyOutput',
              variables: {}
            },
            style: { borderStyle: 'dashed', opacity: 0.8 }
          });
        }
        visibleEdges.push({
          ...edge,
          id: `e-proxy-out-${source}-${target}`,
          target: proxyId,
          targetHandle: 'flow-in',
          style: getEdgeStyle(source, edge.style)
        });
      }
    }
  });

  const onNodesChangeWrapped = useCallback((changes: any) => {
    const filteredChanges = changes.filter((c: any) => !c.id.startsWith('proxy-'));
    onNodesChange(filteredChanges);
  }, [onNodesChange]);

  const onNodeDoubleClick = useCallback((_: any, node: Node) => {
    const hasChildren = nodes.some(n => n.id.startsWith(node.id + '.'));
    const isContainerClass = ['ConcentratorPlant', 'DRSModel', 'Module', 'ProcessingPlant', 'ContinuousFleetLogistics', 'ConcentratorController', 'ConcentratorModel'].includes((node.data as any).class);
    if (hasChildren || isContainerClass) {
      setCurrentParentId(node.id);
      setSelectedNode(null);
    }
  }, [nodes]);

  return (
    <div className="flex w-screen h-screen bg-slate-950 font-sans text-slate-100 overflow-hidden">
      {/* Sidebar palette & controls */}
      <Sidebar 
        onExport={onExport} 
        onImport={onImport} 
        onClear={onClear} 
        onSaveToWorkspace={onSaveToWorkspace}
        onVerifyCompile={onVerifyCompile}
        isSaving={isSaving}
        isCompiling={isCompiling}
      />

      {/* React Flow Editor Grid */}
      <div className="flex-1 h-full relative flex flex-col" ref={reactFlowWrapper}>
        {/* Breadcrumb Navigation bar */}
        {currentParentId && (
          <div className="bg-slate-900 border-b border-slate-800/80 px-4 py-2 flex items-center gap-2 text-xs font-semibold z-10 text-slate-300">
            <button
              onClick={() => setCurrentParentId(null)}
              className="text-sky-400 hover:text-sky-350 transition-colors"
            >
              Root
            </button>
            {currentParentId.split('.').map((part, index, arr) => {
              const path = arr.slice(0, index + 1).join('.');
              const isLast = index === arr.length - 1;
              return (
                <React.Fragment key={path}>
                  <span className="text-slate-600">/</span>
                  {isLast ? (
                    <span className="text-white font-bold">{part}</span>
                  ) : (
                    <button
                      onClick={() => setCurrentParentId(path)}
                      className="text-sky-400 hover:text-sky-350 transition-colors"
                    >
                      {part}
                    </button>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        )}

        <div className="flex-1 h-full relative">
          <ReactFlow
            nodes={visibleNodes}
            edges={visibleEdges}
            onNodesChange={onNodesChangeWrapped}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onNodeDoubleClick={onNodeDoubleClick}
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

      {/* Simulation Playback Toolbar */}
      <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 bg-slate-900/95 backdrop-blur-md border border-slate-800 rounded-xl px-5 py-3 shadow-2xl z-20 flex items-center gap-4 text-xs font-semibold max-w-[90%] md:max-w-2xl select-none">
        <button
          onClick={onRunSimulation}
          className="flex items-center gap-1.5 py-1.5 px-3 rounded-lg bg-sky-600 hover:bg-sky-500 text-white transition-all text-xs font-bold whitespace-nowrap shadow-lg shadow-sky-950/20"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          Run Simulation
        </button>
        <div className="flex items-center gap-1">
          <span className="text-[9px] text-slate-500">Seed</span>
          <input
            type="number"
            min={0}
            value={simSeed}
            onChange={(e) => setSimSeed(parseInt(e.target.value) || 0)}
            className="w-14 bg-slate-800 border border-slate-700 rounded px-1.5 py-1 text-[10px] text-white font-mono text-center focus:outline-none focus:border-sky-600"
          />
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[9px] text-slate-500">Dur(h)</span>
          <input
            type="number"
            min={1}
            max={99999}
            value={maxSimTime}
            onChange={(e) => setMaxSimTime(parseInt(e.target.value) || 99999)}
            className="w-16 bg-slate-800 border border-slate-700 rounded px-1.5 py-1 text-[10px] text-white font-mono text-center focus:outline-none focus:border-sky-600"
          />
        </div>

        {telemetryHistory.length > 0 && (
          <>
            <div className="h-6 w-px bg-slate-800" />
            
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="flex items-center justify-center p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 transition-all text-slate-100"
              title={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? (
                <Pause className="w-4 h-4 text-sky-400 fill-current" />
              ) : (
                <Play className="w-4 h-4 text-emerald-400 fill-current" />
              )}
            </button>

            <button
              onClick={() => {
                setIsPlaying(false);
                setCurrentPlaybackTime(0);
              }}
              className="flex items-center justify-center p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 transition-all text-slate-400 hover:text-white"
              title="Restart"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            <div className="flex items-center gap-2">
              <span className="text-[10px] text-slate-400 whitespace-nowrap">t = {currentPlaybackTime.toFixed(1)}s</span>
              <input
                type="range"
                min={0}
                max={telemetryHistory[telemetryHistory.length - 1]?.time || maxSimTime}
                step={0.1}
                value={currentPlaybackTime}
                onChange={(e) => {
                  setIsPlaying(false);
                  setCurrentPlaybackTime(parseFloat(e.target.value));
                }}
                className="w-32 md:w-40 accent-sky-400 cursor-pointer"
              />
              <span className="text-[10px] text-slate-400 whitespace-nowrap">{(telemetryHistory[telemetryHistory.length - 1]?.time || maxSimTime).toFixed(0)}s</span>
              <span className="text-[10px] text-amber-400/80 whitespace-nowrap font-mono">Day {(currentPlaybackTime / 24).toFixed(1)}</span>
            </div>

            <div className="h-6 w-px bg-slate-800" />

            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-slate-400">Speed</span>
              <select
                value={playbackSpeed}
                onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                className="bg-slate-800 border border-slate-700 rounded px-1.5 py-0.5 text-xs text-white cursor-pointer focus:outline-none"
              >
                <option value={0.5}>0.5x</option>
                <option value={1}>1.0x</option>
                <option value={2}>2.0x</option>
                <option value={5}>5.0x</option>
                <option value={10}>10x</option>
              </select>
            </div>

            {(telemetryEvents.length > 0 || dashboardPng) && (
              <>
                <div className="h-6 w-px bg-slate-800" />
                <button
                  onClick={() => {
                    setShowDashboard(false);
                    setShowEventLog(!showEventLog);
                  }}
                  className={`flex items-center gap-1.5 py-1.5 px-3 rounded-lg border text-xs font-semibold transition-all ${
                    showEventLog 
                      ? 'bg-amber-950/20 border-amber-500/40 text-amber-300' 
                      : 'bg-slate-800 border-slate-700 text-slate-300 hover:text-white'
                  }`}
                >
                  <ListFilter className="w-3.5 h-3.5" />
                  Events ({telemetryEvents.length})
                </button>
                {dashboardPng && (
                  <>
                    <div className="h-6 w-px bg-slate-800" />
                    <button
                      onClick={() => {
                        setShowEventLog(false);
                        setShowDashboard(!showDashboard);
                      }}
                      className={`flex items-center gap-1.5 py-1.5 px-3 rounded-lg border text-xs font-semibold transition-all ${
                        showDashboard 
                          ? 'bg-emerald-950/20 border-emerald-500/40 text-emerald-300' 
                          : 'bg-slate-800 border-slate-700 text-slate-300 hover:text-white'
                      }`}
                    >
                      Dashboard
                    </button>
                  </>
                )}
              </>
            )}
          </>
        )}
      </div>

      {/* Floating Dashboard Panel */}
      {showDashboard && dashboardPng && (
        <div className="absolute top-16 right-6 w-[90vw] max-w-[1200px] bg-slate-900/95 backdrop-blur-md border border-slate-800 rounded-xl p-4 shadow-2xl z-20 flex flex-col max-h-[85vh] select-none text-slate-100">
          <div className="flex justify-between items-center pb-2 mb-3 border-b border-slate-800">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">Diagnostic Dashboard</span>
            <button
              onClick={() => setShowDashboard(false)}
              className="text-slate-400 hover:text-white text-xs font-bold"
            >
              Close
            </button>
          </div>
          <div className="flex-1 overflow-y-auto pr-1">
            <img src={dashboardPng} alt="Diagnostic Dashboard" className="w-full" />
          </div>
        </div>
      )}

      {/* Floating Event Timeline Panel */}
      {showEventLog && telemetryEvents.length > 0 && (
        <div className="absolute top-16 right-6 w-80 bg-slate-900/95 backdrop-blur-md border border-slate-800 rounded-xl p-4 shadow-2xl z-20 flex flex-col max-h-[70vh] select-none text-slate-100">
          <div className="flex justify-between items-center pb-2 mb-3 border-b border-slate-800">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">Event Timeline</span>
            <button
              onClick={() => setShowEventLog(false)}
              className="text-slate-400 hover:text-white text-xs font-bold"
            >
              Close
            </button>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 pr-1">
            {telemetryEvents.map((evt, idx) => {
              const isPast = evt.time <= currentPlaybackTime;
              return (
                <div
                  key={idx}
                  onClick={() => {
                    setIsPlaying(false);
                    setCurrentPlaybackTime(evt.time);
                  }}
                  className={`p-2.5 rounded-lg border cursor-pointer text-left transition-all ${
                    isPast 
                      ? 'bg-slate-950/60 border-slate-800 hover:border-slate-700' 
                      : 'bg-slate-950/10 border-slate-900/20 opacity-50 hover:opacity-80'
                  }`}
                >
                  <div className="flex justify-between items-center mb-1 text-[10px]">
                    <span className="font-mono text-sky-400 font-bold">t = {evt.time.toFixed(2)}s</span>
                    <span className="text-[8px] text-amber-400/70 font-mono">Day {(evt.time / 24).toFixed(2)}</span>
                    <span className="font-sans font-semibold px-1 py-0.2 rounded bg-slate-800 text-slate-300 uppercase text-[8px] tracking-wide">{evt.event_type}</span>
                  </div>
                  <div className="text-xs font-semibold text-slate-200">{evt.source}</div>
                  {evt.details && Object.keys(evt.details).length > 0 && (
                    <div className="mt-1.5 text-[9px] font-mono text-slate-400 border-t border-slate-950/50 pt-1">
                      {Object.entries(evt.details).map(([k, v]) => (
                        <div key={k} className="flex justify-between">
                          <span>{k}:</span>
                          <span className="text-slate-300">{String(v)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

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
