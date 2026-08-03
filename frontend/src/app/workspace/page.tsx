import React from "react";
import { Layout, Plus } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";
import { Button } from "@/components/ui/Button";

export default function WorkspacePage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Workbench Workspace"
        description="Active canvas and multi-pane inspection view for Computer Vision model testing."
        breadcrumbs={["VisionForge", "Workspace"]}
        actions={
          <Button variant="primary" icon={<Plus className="w-4 h-4" />}>
            New Workspace Session
          </Button>
        }
      />

      <EmptyState
        icon={<Layout className="w-6 h-6" />}
        title="Workspace Ready"
        description="This workspace viewport is ready for active session rendering. Future modules (model canvas, dataset inspection, and visualization pipelines) will attach to this canvas."
        action={
          <Button variant="secondary" size="sm">
            View Workspace Guidelines
          </Button>
        }
      />
    </div>
  );
}
