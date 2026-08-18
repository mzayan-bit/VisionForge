"use client";

import React, { useEffect, useState } from "react";
import { Cpu, Search, Filter, MoreHorizontal, DownloadCloud, Box, Hash, HardDrive } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";

// Match the canonical API schema from backend/visionforge/models/metadata.py
interface ModelMetadata {
  name: string;
  version: string;
  author?: string;
  description?: string;
  license: string;
  task: string;
  framework: string;
  supported_devices: string[];
  device_support?: string[];
  source: {
    provider: string;
    repository?: string;
    download_url?: string;
    sha256?: string;
  };
  status: string;
  install_path?: string;
  disk_size_bytes?: number;
  disk_size_mb?: number;
  installed_at?: string;
  last_used_at?: string;
  updated_at?: string;
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchModels() {
      try {
        const response = await fetch("/api/v1/models");
        const json = await response.json();
        if (json.success && json.data) {
          const rawModels: any[] = json.data.models || [];
          const normalizedModels: ModelMetadata[] = rawModels.map((m: any) => ({
            ...m,
            supported_devices:
              Array.isArray(m.supported_devices) && m.supported_devices.length > 0
                ? m.supported_devices
                : Array.isArray(m.device_support) && m.device_support.length > 0
                ? m.device_support
                : ["cpu", "cuda", "mps"],
            status: m.status || "installed",
            source: m.source || { provider: "local" },
          }));
          setModels(normalizedModels);
        }
      } catch (error) {
        console.error("Failed to fetch models:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchModels();
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "installed":
        return "success";
      case "installing":
        return "warning";
      case "error":
        return "error";
      default:
        return "default";
    }
  };

  const getDeviceIcon = (device: string) => {
    if (device === "cpu") return <Cpu className="w-3 h-3 text-secondary-500 mr-1" />;
    return <Cpu className="w-3 h-3 text-primary-500 mr-1" />;
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Models"
        description="Manage and deploy computer vision architectures."
        breadcrumbs={["VisionForge", "Models"]}
        actions={
          <Button variant="primary" icon={<DownloadCloud className="w-4 h-4" />}>
            Browse Marketplace
          </Button>
        }
      />

      {!loading && models.length === 0 ? (
        <EmptyState
          icon={<Box className="w-6 h-6" />}
          title="No models found"
          description="Your model library is currently empty. Start by browsing the model marketplace or importing your own weights."
          action={
            <div className="flex space-x-3">
              <Button variant="primary" size="sm">
                Browse Marketplace
              </Button>
              <Button variant="secondary" size="sm">
                Import Model Weights
              </Button>
            </div>
          }
        />
      ) : (
        <Card className="flex flex-col">
          {/* Toolbar */}
          <div className="p-4 border-b border-surface-700/50 flex justify-between items-center bg-surface-900/20 backdrop-blur-sm">
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-surface-400" />
              <input 
                type="text" 
                placeholder="Search models..." 
                className="pl-9 pr-4 py-1.5 bg-surface-950 border border-surface-700 rounded-md text-sm text-surface-100 placeholder-surface-500 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/50 w-64 transition-all"
              />
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="secondary" size="sm" icon={<Filter className="w-4 h-4" />}>
                Filter
              </Button>
            </div>
          </div>

          {/* Table */}
          <div className="w-full overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-surface-400 bg-surface-900/40 uppercase">
                <tr>
                  <th className="px-6 py-4 font-medium tracking-wider">Model Name</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Status</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Version</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Task</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Disk Usage</th>
                  <th className="px-6 py-4 font-medium tracking-wider">Device Support</th>
                  <th className="px-6 py-4 font-medium tracking-wider text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-800/50">
                {models.map((model) => (
                  <tr key={model.name} className="hover:bg-surface-800/20 transition-colors group">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded bg-primary-900/30 border border-primary-500/20 flex items-center justify-center">
                          <Box className="w-4 h-4 text-primary-400" />
                        </div>
                        <div>
                          <div className="font-medium text-surface-50">{model.name}</div>
                          <div className="text-xs text-surface-500 capitalize">{model.framework} • {model.source?.provider || "local"}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <Badge variant={getStatusColor(model.status) as any}>{model.status}</Badge>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center text-surface-300 font-mono text-xs">
                        <Hash className="w-3 h-3 mr-1 text-surface-500" />
                        {model.version}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-surface-200 capitalize">{model.task.replace('_', ' ')}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-surface-300">
                      <div className="flex items-center">
                        <HardDrive className="w-3 h-3 mr-1.5 text-surface-500" />
                        {model.disk_size_mb ? `${model.disk_size_mb.toFixed(1)} MB` : 'Unknown'}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        {model.supported_devices && model.supported_devices.length > 0 ? (
                          model.supported_devices.map((device) => (
                            <span key={device} className="flex items-center text-xs px-2 py-0.5 rounded-full bg-surface-800 border border-surface-700 text-surface-300 uppercase">
                              {getDeviceIcon(device)}
                              {device}
                            </span>
                          ))
                        ) : (
                          <span className="text-xs text-surface-500 italic">None</span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right">
                      <div className="flex items-center justify-end space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Button variant="primary" size="sm">Deploy</Button>
                        <Button variant="secondary" size="sm" className="px-2">
                          <MoreHorizontal className="w-4 h-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
