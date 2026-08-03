import React from "react";
import Link from "next/link";
import {
  Terminal,
  ArrowRight,
  Shield,
  Layers,
  Activity,
  Cpu,
  BookOpen,
  Plus,
  Clock,
  Sparkles,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { Card, CardBody } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";

export default function HomePage() {
  return (
    <div className="space-y-8">
      {/* Page Header */}
      <PageHeader
        title="Workbench Home"
        description="VisionForge Computer Vision Workbench Core Architecture & Engineering Workspace"
        breadcrumbs={["VisionForge", "Home"]}
        actions={
          <Link href="/workspace">
            <Button variant="primary" icon={<Plus className="w-4 h-4" />}>
              Open Workspace
            </Button>
          </Link>
        }
      />

      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-xl border border-white/10 bg-gradient-to-r from-blue-950/40 via-neutral-900/60 to-neutral-900/40 p-6 md:p-8 backdrop-blur-md">
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="flex items-center gap-2">
            <Badge variant="info" dot size="sm">
              PHASE 1 ACTIVE
            </Badge>
            <span className="text-xs font-mono text-neutral-400">
              Core Backend & Frontend Foundation Established
            </span>
          </div>

          <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-white font-geist">
            Welcome to <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">VisionForge</span>
          </h1>

          <p className="text-sm md:text-base text-neutral-300 leading-relaxed font-normal">
            A high-performance Computer Vision Workbench engineered around decoupled FastAPI core services, 
            standardized adapter interfaces, and a high DX Next.js workbench interface.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Link href="/workspace">
              <Button variant="primary" icon={<ArrowRight className="w-4 h-4" />} iconPosition="right">
                Launch Workspace
              </Button>
            </Link>
            <Link href="/documentation">
              <Button variant="secondary" icon={<BookOpen className="w-4 h-4" />}>
                Read Architecture Docs
              </Button>
            </Link>
          </div>
        </div>

        {/* Ambient Glow Graphic */}
        <div className="absolute -right-20 -top-20 w-80 h-80 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />
      </div>

      {/* Quick Action Tiles */}
      <div>
        <SectionHeader title="Quick Actions & Operations" badge="3 TILES" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card hoverable className="group">
            <CardBody className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 group-hover:border-blue-400/40 transition-colors">
                <Layers className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-white font-geist">Workbench Engine</h3>
              <p className="text-xs text-neutral-400 leading-relaxed">
                Explore the modular workspace layout with 5-pane inspection and dark mode controls.
              </p>
              <Link href="/workspace" className="inline-flex items-center gap-1.5 text-xs text-blue-400 font-medium hover:text-blue-300">
                Go to Workspace <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </CardBody>
          </Card>

          <Card hoverable className="group">
            <CardBody className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 group-hover:border-emerald-400/40 transition-colors">
                <Cpu className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-white font-geist">System Diagnostics</h3>
              <p className="text-xs text-neutral-400 leading-relaxed">
                Inspect FastAPI `/api/v1/system/info` telemetry, registered routes, and runtime health.
              </p>
              <Link href="/settings" className="inline-flex items-center gap-1.5 text-xs text-emerald-400 font-medium hover:text-emerald-300">
                View Diagnostics <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </CardBody>
          </Card>

          <Card hoverable className="group">
            <CardBody className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400 group-hover:border-purple-400/40 transition-colors">
                <BookOpen className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-white font-geist">Platform Documentation</h3>
              <p className="text-xs text-neutral-400 leading-relaxed">
                Read system specifications for backend configuration, response contracts, and UI guidelines.
              </p>
              <Link href="/documentation" className="inline-flex items-center gap-1.5 text-xs text-purple-400 font-medium hover:text-purple-300">
                Open Documentation <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </CardBody>
          </Card>
        </div>
      </div>

      {/* System Status & Recent Activity Split */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Development Phase Summary (2 cols) */}
        <div className="lg:col-span-2 space-y-4">
          <SectionHeader title="Development Milestones" badge="CURRENT" />
          <Card>
            <CardBody className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-blue-400" />
                  <h4 className="text-sm font-semibold text-white font-geist">
                    Phase 1: Foundation & Core Architecture
                  </h4>
                </div>
                <Badge variant="success" size="sm">COMPLETED</Badge>
              </div>

              <div className="space-y-2 text-xs text-neutral-300">
                <div className="flex items-start gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0" />
                  <span><strong>Backend Core:</strong> Pydantic v2 Settings, ANSI structured logging, lifecycle handlers, dependency injection, and centralized exception handling.</span>
                </div>
                <div className="flex items-start gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0" />
                  <span><strong>Unified Response Protocol:</strong> Standardized <code>APIResponse[T]</code> JSON envelope for all backend routes.</span>
                </div>
                <div className="flex items-start gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0" />
                  <span><strong>Frontend Shell:</strong> Next.js App Router, Tailwind CSS dark mode tokens, 5-pane layout, command palette, and reusable UI components.</span>
                </div>
              </div>
            </CardBody>
          </Card>
        </div>

        {/* Recent Activity Feed Placeholder (1 col) */}
        <div className="space-y-4">
          <SectionHeader title="Recent Activity" badge="LOG" />
          <Card>
            <CardBody className="space-y-3 font-mono text-xs">
              <div className="flex items-start gap-3 pb-3 border-b border-white/10">
                <Clock className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
                <div>
                  <p className="text-neutral-200 font-medium">Frontend layout & shell updated</p>
                  <span className="text-[10px] text-neutral-500">Just now</span>
                </div>
              </div>

              <div className="flex items-start gap-3 pb-3 border-b border-white/10">
                <Shield className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <p className="text-neutral-200 font-medium">Backend architecture committed</p>
                  <span className="text-[10px] text-neutral-500">1 hour ago</span>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <Activity className="w-4 h-4 text-purple-400 shrink-0 mt-0.5" />
                <div>
                  <p className="text-neutral-200 font-medium">Repository main initialized</p>
                  <span className="text-[10px] text-neutral-500">Today</span>
                </div>
              </div>
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  );
}
