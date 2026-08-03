import React from "react";
import { Settings, Server, Shield, Cpu, Activity } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings & System Diagnostics"
        description="Workbench platform configuration, theme preferences, and FastAPI telemetry."
        breadcrumbs={["VisionForge", "Settings"]}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Settings className="w-4 h-4 text-blue-400" />
              <h3 className="text-sm font-semibold text-white font-geist">Theme & Preference Controls</h3>
            </div>
          </CardHeader>
          <CardBody className="space-y-4 text-xs">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-white">Default Color Mode</p>
                <p className="text-neutral-400 text-[11px]">Obsidian dark mode protocol</p>
              </div>
              <Badge variant="info" size="sm">Dark Mode First</Badge>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-white/10">
              <div>
                <p className="font-medium text-white">Interface Spacing Density</p>
                <p className="text-neutral-400 text-[11px]">Engineering grade 4px rhythm</p>
              </div>
              <Badge variant="neutral" size="sm">Compact</Badge>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-semibold text-white font-geist">FastAPI Telemetry Diagnostics</h3>
            </div>
          </CardHeader>
          <CardBody className="space-y-3 text-xs font-mono">
            <div className="flex justify-between text-neutral-400">
              <span>FastAPI Target Host:</span>
              <span className="text-neutral-200">http://0.0.0.0:8000</span>
            </div>
            <div className="flex justify-between text-neutral-400">
              <span>API Version Prefix:</span>
              <span className="text-emerald-400">/api/v1</span>
            </div>
            <div className="flex justify-between text-neutral-400">
              <span>Health Endpoint:</span>
              <span className="text-blue-400">/api/v1/health</span>
            </div>
            <div className="flex justify-between text-neutral-400">
              <span>System Metadata Endpoint:</span>
              <span className="text-blue-400">/api/v1/system/info</span>
            </div>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
