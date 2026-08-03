import React from "react";
import { FlaskConical, Plus } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";
import { Button } from "@/components/ui/Button";

export default function ExperimentsPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Experiment Tracker"
        description="Run tracking, hyperparameter configuration, and reproducible experiment logs."
        breadcrumbs={["VisionForge", "Experiments"]}
        actions={
          <Button variant="primary" icon={<Plus className="w-4 h-4" />}>
            New Experiment Run
          </Button>
        }
      />

      <EmptyState
        icon={<FlaskConical className="w-6 h-6" />}
        title="No Active Experiments"
        description="The Experiment Tracker module will organize hyperparameter trials, artifact logging, and reproducibility runs."
        action={
          <Button variant="secondary" size="sm">
            View Experiment Schema
          </Button>
        }
      />
    </div>
  );
}
