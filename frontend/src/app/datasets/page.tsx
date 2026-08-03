import React from "react";
import { Database, Plus } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";
import { Button } from "@/components/ui/Button";

export default function DatasetsPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Dataset Registry"
        description="Dataset metadata, split management, and visual data sample indexing."
        breadcrumbs={["VisionForge", "Datasets"]}
        actions={
          <Button variant="primary" icon={<Plus className="w-4 h-4" />}>
            Register Dataset Manifest
          </Button>
        }
      />

      <EmptyState
        icon={<Database className="w-6 h-6 text-emerald-400" />}
        title="Dataset Registry Ready"
        description="The Dataset Registry module will index computer vision dataset manifests, annotation formats (COCO, YOLO, Pascal VOC), and data splits."
        action={
          <Button variant="secondary" size="sm">
            View Dataset Manifest Spec
          </Button>
        }
      />
    </div>
  );
}
