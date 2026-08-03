import React from "react";
import { Cpu, Plus } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";
import { Button } from "@/components/ui/Button";

export default function ModelsPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Registry"
        description="Catalog and metadata manager for vision foundation models."
        breadcrumbs={["VisionForge", "Models"]}
        actions={
          <Button variant="primary" icon={<Plus className="w-4 h-4" />}>
            Register Model Specification
          </Button>
        }
      />

      <EmptyState
        icon={<Cpu className="w-6 h-6" />}
        title="No Models Registered"
        description="The Model Registry module will provide standardized model specifications, version tracking, and adapter definitions in upcoming phases."
        action={
          <Button variant="secondary" size="sm">
            Read Model Specification Docs
          </Button>
        }
      />
    </div>
  );
}
