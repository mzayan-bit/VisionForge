import React from "react";
import { BarChart2, Plus } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";
import { Button } from "@/components/ui/Button";

export default function BenchmarksPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Benchmarking Engine"
        description="Standardized metrics evaluation and comparison suite for vision pipelines."
        breadcrumbs={["VisionForge", "Benchmarks"]}
        actions={
          <Button variant="primary" icon={<Plus className="w-4 h-4" />}>
            Configure Benchmark Suite
          </Button>
        }
      />

      <EmptyState
        icon={<BarChart2 className="w-6 h-6" />}
        title="Benchmarking Suite Ready"
        description="The Benchmarking Engine will evaluate speed, accuracy, and resource utilization across registered models and hardware target adapters."
        action={
          <Button variant="secondary" size="sm">
            Explore Metric Protocols
          </Button>
        }
      />
    </div>
  );
}
