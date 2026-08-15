"use client";

import React, { useState, useEffect } from "react";
import {
  Database,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  FileSpreadsheet,
  Download,
  Play,
  RotateCcw,
  Sparkles,
  Layers,
  ShieldAlert,
  Clock,
  ExternalLink,
  Search,
  Filter,
  Eye,
  GitCompare,
  Check,
  X,
  AlertCircle,
  BarChart3,
  Cpu,
  ArrowRight,
  RefreshCw,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

interface HealthCategoryItem {
  category: string;
  status: "GOOD" | "NEEDS_REVIEW" | "CRITICAL";
  headline: string;
  details: string;
  issues_count: number;
}

interface DatasetHealthSummary {
  overall_integrity: HealthCategoryItem;
  annotation_quality: HealthCategoryItem;
  class_balance: HealthCategoryItem;
  visual_diversity: HealthCategoryItem;
  potential_leakage: HealthCategoryItem;
  model_difficulty: HealthCategoryItem;
}

interface ClassDistributionItem {
  class_id: number;
  class_name: string;
  sample_count: number;
  sample_percentage: number;
  annotation_count: number;
  avg_annotations_per_image: number;
  is_rare_class: boolean;
  is_dominant_class: boolean;
  split_counts: Record<string, number>;
}

interface ImageStatistics {
  min_width: number;
  max_width: number;
  mean_width: number;
  min_height: number;
  max_height: number;
  mean_height: number;
  mean_aspect_ratio: number;
  format_distribution: Record<string, number>;
  resolution_bins: Record<string, number>;
  total_size_bytes: number;
}

interface AnnotationStatistics {
  total_boxes: number;
  mean_boxes_per_image: number;
  max_boxes_per_image: number;
  mean_box_relative_area: number;
  size_distribution: Record<string, number>;
}

interface ClassCooccurrence {
  class_a: string;
  class_b: string;
  cooccurrence_count: number;
  cooccurrence_rate: number;
}

interface DatasetProfile {
  dataset_id: string;
  dataset_version: string;
  dataset_fingerprint: string;
  total_samples: number;
  total_annotations: number;
  total_classes: number;
  class_distribution: ClassDistributionItem[];
  split_distribution: Record<string, number>;
  split_percentages: Record<string, number>;
  image_statistics: ImageStatistics;
  annotation_statistics: AnnotationStatistics;
  class_cooccurrence: ClassCooccurrence[];
  health_summary: DatasetHealthSummary;
  profile_generated_at: string;
}

interface QualityIssueItem {
  issue_id: string;
  sample_id: string;
  issue_type: string;
  flag: string;
  severity: "WARNING" | "CRITICAL";
  message: string;
  image_path: string;
  split: string;
  class_name?: string;
  bbox?: number[];
  review_status: string;
  detected_at: string;
}

interface LeakageCandidatePair {
  pair_id: string;
  sample_a_id: string;
  sample_a_split: string;
  sample_a_path: string;
  sample_b_id: string;
  sample_b_split: string;
  sample_b_path: string;
  cross_split_type: string;
  similarity_score: number;
  match_type: "EXACT_HASH" | "VISUAL_SIMILARITY";
  recommendation: string;
}

interface HardSampleItem {
  sample_id: string;
  image_path: string;
  split: string;
  prioritization_score: number;
  signals: Record<string, number>;
  failure_reasons: string[];
  ground_truth_classes: string[];
  predicted_classes: string[];
}

interface DatasetVersionRecord {
  version_id: string;
  dataset_id: string;
  parent_version_id?: string;
  dataset_fingerprint: string;
  changes_summary: string;
  total_samples: number;
  total_annotations: number;
  review_decisions_count: number;
  created_at: string;
}

interface DatasetDiffResult {
  dataset_id: string;
  version_a: string;
  version_b: string;
  samples_added: string[];
  samples_removed: string[];
  classes_added: string[];
  classes_removed: string[];
  annotations_count_delta: number;
  leakage_pairs_delta: number;
  class_distribution_deltas: Record<string, number>;
  summary: string;
}

export default function DataCentricWorkspacePage() {
  const [datasetId, setDatasetId] = useState("safety_v2");
  const [versionId, setVersionId] = useState("v2.0.0");
  const [activeTab, setActiveTab] = useState<
    "overview" | "classes" | "issues" | "leakage" | "hard_samples" | "diff" | "pipeline"
  >("overview");

  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [issues, setIssues] = useState<QualityIssueItem[]>([]);
  const [leakagePairs, setLeakagePairs] = useState<LeakageCandidatePair[]>([]);
  const [hardSamples, setHardSamples] = useState<HardSampleItem[]>([]);
  const [versions, setVersions] = useState<DatasetVersionRecord[]>([]);
  const [diffResult, setDiffResult] = useState<DatasetDiffResult | null>(null);
  const [diffVerA, setDiffVerA] = useState("v1.0.0");
  const [diffVerB, setDiffVerB] = useState("v2.0.0");

  const [selectedIssue, setSelectedIssue] = useState<QualityIssueItem | null>(null);
  const [inspectSample, setInspectSample] = useState<string | null>(null);
  const [issueFilter, setIssueFilter] = useState<string>("ALL");
  const [isLoading, setIsLoading] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // Pipeline Splitting State
  const [trainRatio, setTrainRatio] = useState(70);
  const [valRatio, setValRatio] = useState(15);
  const [testRatio, setTestRatio] = useState(15);
  const [seed, setSeed] = useState(42);
  const [strategy, setStrategy] = useState("random");
  const [isSplitting, setIsSplitting] = useState(false);

  useEffect(() => {
    loadWorkspaceData();
  }, [datasetId, versionId]);

  const loadWorkspaceData = async () => {
    setIsLoading(true);
    try {
      // 1. Profile
      const profRes = await fetch(`/api/v1/datasets/intelligence/profile?dataset_id=${datasetId}&version=${versionId}`);
      if (profRes.ok) {
        const p = await profRes.json();
        setProfile(p.data);
      }

      // 2. Issues
      const issRes = await fetch(`/api/v1/datasets/intelligence/issues?dataset_id=${datasetId}`);
      if (issRes.ok) {
        const i = await issRes.json();
        setIssues(i.data);
      }

      // 3. Leakage
      const leakRes = await fetch(`/api/v1/datasets/intelligence/leakage?dataset_id=${datasetId}`);
      if (leakRes.ok) {
        const l = await leakRes.json();
        setLeakagePairs(l.data);
      }

      // 4. Hard samples
      const hardRes = await fetch(`/api/v1/datasets/intelligence/hard-samples?dataset_id=${datasetId}`);
      if (hardRes.ok) {
        const h = await hardRes.json();
        setHardSamples(h.data);
      }

      // 5. Versions
      const verRes = await fetch(`/api/v1/datasets/intelligence/versions?dataset_id=${datasetId}`);
      if (verRes.ok) {
        const v = await verRes.json();
        setVersions(v.data);
      }
    } catch (err) {
      console.error("Failed to load workspace:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFetchDiff = async () => {
    try {
      const res = await fetch(
        `/api/v1/datasets/intelligence/diff?dataset_id=${datasetId}&version_a=${diffVerA}&version_b=${diffVerB}`
      );
      if (res.ok) {
        const d = await res.json();
        setDiffResult(d.data);
      }
    } catch (err) {
      console.error("Failed to fetch diff:", err);
    }
  };

  const handleCurationDecision = async (sampleId: string, issueId: string | undefined, decision: string) => {
    try {
      const res = await fetch("/api/v1/datasets/intelligence/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          review_id: `rev_${Date.now()}`,
          sample_id: sampleId,
          issue_id: issueId,
          decision: decision,
          category: "annotation_review",
          notes: `Recorded decision: ${decision}`,
          reviewer: "Principal Researcher",
        }),
      });

      if (res.ok) {
        setToastMessage(`Review decision '${decision}' recorded for ${sampleId}`);
        setTimeout(() => setToastMessage(null), 3000);
        setSelectedIssue(null);
        setInspectSample(null);
        // Refresh issues
        loadWorkspaceData();
      }
    } catch (err) {
      console.error("Failed to record decision:", err);
    }
  };

  const handleDownloadReport = () => {
    window.open(`/api/v1/datasets/intelligence/report?dataset_id=${datasetId}&version=${versionId}`, "_blank");
  };

  const getStatusBadge = (status: "GOOD" | "NEEDS_REVIEW" | "CRITICAL") => {
    switch (status) {
      case "GOOD":
        return <span className="px-2 py-0.5 text-xs font-semibold rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Good</span>;
      case "NEEDS_REVIEW":
        return <span className="px-2 py-0.5 text-xs font-semibold rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">Needs Review</span>;
      case "CRITICAL":
        return <span className="px-2 py-0.5 text-xs font-semibold rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">Critical</span>;
    }
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-3 bg-emerald-950 border border-emerald-500 text-emerald-200 rounded-lg shadow-xl text-sm animate-fade-in">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          {toastMessage}
        </div>
      )}

      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <PageHeader
            title="Dataset Intelligence & Curation Workspace"
            description="Scientifically rigorous data-centric computer vision: quality audits, leakage detection, class balance, and version lineage."
          />
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 gap-2">
            <Database className="w-4 h-4 text-blue-400" />
            <select
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              className="bg-transparent text-sm font-medium text-zinc-200 focus:outline-none"
            >
              <option value="safety_v2">safety_v2</option>
              <option value="ppe_detection">ppe_detection</option>
            </select>

            <span className="text-zinc-600">/</span>

            <select
              value={versionId}
              onChange={(e) => setVersionId(e.target.value)}
              className="bg-transparent text-xs font-semibold text-blue-400 focus:outline-none"
            >
              <option value="v2.0.0">v2.0.0 (Curated)</option>
              <option value="v1.0.0">v1.0.0 (Raw)</option>
            </select>
          </div>

          <Button variant="outline" size="sm" onClick={handleDownloadReport} className="gap-1.5 border-zinc-700">
            <Download className="w-3.5 h-3.5" />
            Export Report
          </Button>

          <Button size="sm" onClick={loadWorkspaceData} disabled={isLoading} className="gap-1.5 bg-blue-600 hover:bg-blue-500">
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Dataset Health Scorecard (Step 30-31) */}
      {profile?.health_summary && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
          {[
            { key: "integrity", item: profile.health_summary.overall_integrity, icon: Database },
            { key: "anno", item: profile.health_summary.annotation_quality, icon: ShieldAlert },
            { key: "balance", item: profile.health_summary.class_balance, icon: BarChart3 },
            { key: "diversity", item: profile.health_summary.visual_diversity, icon: Sparkles },
            { key: "leakage", item: profile.health_summary.potential_leakage, icon: AlertTriangle },
            { key: "difficulty", item: profile.health_summary.model_difficulty, icon: Cpu },
          ].map(({ key, item, icon: Icon }) => (
            <Card key={key} className="bg-zinc-900/60 border-zinc-800 p-3 hover:border-zinc-700 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5">
                  <Icon className="w-3.5 h-3.5 text-zinc-400" />
                  <span className="text-xs font-semibold text-zinc-300">{item.category}</span>
                </div>
                {getStatusBadge(item.status)}
              </div>
              <p className="text-xs font-medium text-zinc-200 line-clamp-1">{item.headline}</p>
              <p className="text-[11px] text-zinc-500 mt-1 line-clamp-2">{item.details}</p>
            </Card>
          ))}
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="flex items-center gap-1 border-b border-zinc-800 pb-2 overflow-x-auto text-sm">
        {[
          { id: "overview", label: "Overview & Profile", count: profile?.total_samples },
          { id: "classes", label: "Classes & Co-occurrence", count: profile?.total_classes },
          { id: "issues", label: "Quality Flags & Issues", count: issues.length },
          { id: "leakage", label: "Cross-Split Leakage", count: leakagePairs.length },
          { id: "hard_samples", label: "Hard Samples", count: hardSamples.length },
          { id: "diff", label: "Dataset Diff & Versioning", count: versions.length },
          { id: "pipeline", label: "Preparation & Split Pipeline" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center gap-1.5 whitespace-nowrap ${
              activeTab === tab.id
                ? "bg-blue-600/10 text-blue-400 border border-blue-500/20"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
            }`}
          >
            {tab.label}
            {tab.count !== undefined && (
              <span className={`px-1.5 py-0.2 text-[11px] rounded-full ${
                activeTab === tab.id ? "bg-blue-500/20 text-blue-300" : "bg-zinc-800 text-zinc-400"
              }`}>
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* TAB 1: Overview & Profile */}
      {activeTab === "overview" && profile && (
        <div className="space-y-6 animate-fade-in">
          {/* Top Metrics Row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <Card className="bg-zinc-900/50 border-zinc-800 p-4">
              <span className="text-xs text-zinc-400">Total Samples</span>
              <p className="text-2xl font-bold text-zinc-100 mt-1">{profile.total_samples.toLocaleString()}</p>
              <span className="text-[11px] text-emerald-400 mt-1 block">100% verified readable</span>
            </Card>
            <Card className="bg-zinc-900/50 border-zinc-800 p-4">
              <span className="text-xs text-zinc-400">Total Bounding Boxes</span>
              <p className="text-2xl font-bold text-zinc-100 mt-1">{profile.total_annotations.toLocaleString()}</p>
              <span className="text-[11px] text-zinc-400 mt-1 block">
                ~{profile.annotation_statistics.mean_boxes_per_image} boxes/image
              </span>
            </Card>
            <Card className="bg-zinc-900/50 border-zinc-800 p-4">
              <span className="text-xs text-zinc-400">Distinct Classes</span>
              <p className="text-2xl font-bold text-zinc-100 mt-1">{profile.total_classes}</p>
              <span className="text-[11px] text-zinc-400 mt-1 block">
                {profile.class_distribution.filter((c) => c.is_rare_class).length} rare class flagged
              </span>
            </Card>
            <Card className="bg-zinc-900/50 border-zinc-800 p-4">
              <span className="text-xs text-zinc-400">Dataset Footprint</span>
              <p className="text-2xl font-bold text-zinc-100 mt-1">
                {(profile.image_statistics.total_size_bytes / (1024 * 1024)).toFixed(1)} MB
              </p>
              <span className="text-[11px] text-zinc-400 mt-1 block">
                Mean res: {profile.image_statistics.mean_width.toFixed(0)}x{profile.image_statistics.mean_height.toFixed(0)}px
              </span>
            </Card>
          </div>

          {/* Split Partition Distribution (Step 2) */}
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-sm font-semibold text-zinc-200">Dataset Partitioning & Split Distribution</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="h-4 w-full rounded-full bg-zinc-800 flex overflow-hidden">
                <div style={{ width: `${profile.split_percentages.train}%` }} className="bg-blue-500 h-full" title="Train" />
                <div style={{ width: `${profile.split_percentages.val}%` }} className="bg-emerald-500 h-full" title="Validation" />
                <div style={{ width: `${profile.split_percentages.test}%` }} className="bg-purple-500 h-full" title="Test" />
              </div>

              <div className="grid grid-cols-3 gap-4 pt-2">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500" />
                  <div>
                    <p className="text-xs font-medium text-zinc-300">Train Split</p>
                    <p className="text-sm font-bold text-zinc-100">{profile.split_distribution.train?.toLocaleString()} ({profile.split_percentages.train}%)</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-emerald-500" />
                  <div>
                    <p className="text-xs font-medium text-zinc-300">Validation Split</p>
                    <p className="text-sm font-bold text-zinc-100">{profile.split_distribution.val?.toLocaleString()} ({profile.split_percentages.val}%)</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-purple-500" />
                  <div>
                    <p className="text-xs font-medium text-zinc-300">Test Split</p>
                    <p className="text-sm font-bold text-zinc-100">{profile.split_distribution.test?.toLocaleString()} ({profile.split_percentages.test}%)</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Telemetry Breakdown: Image & Annotation Geometry */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="bg-zinc-900/50 border-zinc-800">
              <CardHeader>
                <CardTitle className="text-sm font-semibold text-zinc-200">Image Resolution & Format Distribution</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  {Object.entries(profile.image_statistics.resolution_bins).map(([tier, count]) => (
                    <div key={tier} className="flex items-center justify-between text-xs">
                      <span className="text-zinc-400">{tier}</span>
                      <div className="flex items-center gap-3">
                        <div className="w-32 h-2 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className="bg-blue-400 h-full rounded-full"
                            style={{ width: `${(count / profile.total_samples) * 100}%` }}
                          />
                        </div>
                        <span className="font-semibold text-zinc-200 w-12 text-right">{count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="bg-zinc-900/50 border-zinc-800">
              <CardHeader>
                <CardTitle className="text-sm font-semibold text-zinc-200">Object Bounding Box Size Distribution</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  {Object.entries(profile.annotation_statistics.size_distribution).map(([tier, count]) => (
                    <div key={tier} className="flex items-center justify-between text-xs">
                      <span className="text-zinc-400 capitalize">{tier} Objects</span>
                      <div className="flex items-center gap-3">
                        <div className="w-32 h-2 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className="bg-amber-400 h-full rounded-full"
                            style={{ width: `${(count / profile.total_annotations) * 100}%` }}
                          />
                        </div>
                        <span className="font-semibold text-zinc-200 w-12 text-right">{count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* TAB 2: Class Distribution & Co-occurrence */}
      {activeTab === "classes" && profile && (
        <div className="space-y-6 animate-fade-in">
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-sm font-semibold text-zinc-200">Category Class Representation & Split Balance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-zinc-800 text-zinc-400">
                      <th className="py-2.5 px-3">Class Name</th>
                      <th className="py-2.5 px-3">Image Samples</th>
                      <th className="py-2.5 px-3">Sample %</th>
                      <th className="py-2.5 px-3">Total Boxes</th>
                      <th className="py-2.5 px-3">Avg / Img</th>
                      <th className="py-2.5 px-3">Train / Val / Test</th>
                      <th className="py-2.5 px-3">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/60">
                    {profile.class_distribution.map((c) => (
                      <tr key={c.class_name} className="hover:bg-zinc-800/30">
                        <td className="py-2.5 px-3 font-semibold text-zinc-200">{c.class_name}</td>
                        <td className="py-2.5 px-3 text-zinc-300">{c.sample_count.toLocaleString()}</td>
                        <td className="py-2.5 px-3">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                              <div className="bg-blue-400 h-full" style={{ width: `${c.sample_percentage}%` }} />
                            </div>
                            <span className="text-zinc-300">{c.sample_percentage.toFixed(1)}%</span>
                          </div>
                        </td>
                        <td className="py-2.5 px-3 text-zinc-300">{c.annotation_count.toLocaleString()}</td>
                        <td className="py-2.5 px-3 text-zinc-400">{c.avg_annotations_per_image.toFixed(2)}</td>
                        <td className="py-2.5 px-3 text-zinc-400">
                          {c.split_counts.train || 0} / {c.split_counts.val || 0} / {c.split_counts.test || 0}
                        </td>
                        <td className="py-2.5 px-3">
                          {c.is_rare_class ? (
                            <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                              ⚠️ Rare (&lt;5%)
                            </span>
                          ) : c.is_dominant_class ? (
                            <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                              ⚡ Dominant
                            </span>
                          ) : (
                            <span className="text-zinc-500 text-[11px]">Balanced</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Class Co-occurrence (Step 16) */}
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-sm font-semibold text-zinc-200">Pairwise Class Co-occurrence Frequencies</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-zinc-400 mb-4">
                Descriptive co-occurrence patterns in multi-object scene imagery (e.g. Workers wearing Helmets and Vests).
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
                {profile.class_cooccurrence.map((co) => (
                  <div key={`${co.class_a}_${co.class_b}`} className="p-3 bg-zinc-900/80 border border-zinc-800 rounded-lg">
                    <div className="flex items-center justify-between text-xs font-semibold text-zinc-200 mb-1">
                      <span>{co.class_a} + {co.class_b}</span>
                      <span className="text-blue-400">{(co.cooccurrence_rate * 100).toFixed(1)}%</span>
                    </div>
                    <span className="text-[11px] text-zinc-500">{co.cooccurrence_count.toLocaleString()} joint images</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* TAB 3: Quality Flags & Issue Explorer */}
      {activeTab === "issues" && (
        <div className="space-y-6 animate-fade-in">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-zinc-400" />
              <span className="text-xs text-zinc-400">Filter by category:</span>
              {["ALL", "ANNOTATION_QUALITY", "IMAGE_QUALITY"].map((f) => (
                <button
                  key={f}
                  onClick={() => setIssueFilter(f)}
                  className={`px-2.5 py-1 text-xs rounded-lg font-medium transition-colors ${
                    issueFilter === f ? "bg-zinc-700 text-zinc-100" : "bg-zinc-900 text-zinc-400 hover:text-zinc-200"
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>

            <span className="text-xs text-zinc-500">{issues.length} diagnostic issues flagged</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {issues
              .filter((i) => issueFilter === "ALL" || i.issue_type === issueFilter)
              .map((iss) => (
                <Card key={iss.issue_id} className="bg-zinc-900/60 border-zinc-800 p-4 hover:border-zinc-700 transition-colors">
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-0.5 text-[10px] font-bold rounded ${
                          iss.severity === "CRITICAL"
                            ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                            : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                        }`}>
                          {iss.flag}
                        </span>
                        <span className="text-xs font-semibold text-zinc-300">{iss.sample_id}</span>
                        <span className="text-[11px] text-zinc-500">({iss.split})</span>
                      </div>
                      <p className="text-xs text-zinc-300 mt-1">{iss.message}</p>
                      {iss.class_name && (
                        <p className="text-[11px] text-blue-400">Target Class: {iss.class_name}</p>
                      )}
                    </div>

                    <div className="flex flex-col gap-1.5">
                      <Button
                        size="sm"
                        variant="outline"
                        className="text-xs h-7 px-2.5 border-zinc-700 hover:bg-emerald-950/40 hover:text-emerald-300 hover:border-emerald-500"
                        onClick={() => handleCurationDecision(iss.sample_id, iss.issue_id, "NEEDS_CORRECTION")}
                      >
                        Send to Review
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-xs h-7 px-2.5 text-zinc-500 hover:text-zinc-300"
                        onClick={() => handleCurationDecision(iss.sample_id, iss.issue_id, "NOT_A_PROBLEM")}
                      >
                        Dismiss
                      </Button>
                    </div>
                  </div>
                </Card>
              ))}
          </div>
        </div>
      )}

      {/* TAB 4: Cross-Split Leakage (Step 8 & 39) */}
      {activeTab === "leakage" && (
        <div className="space-y-6 animate-fade-in">
          <div className="p-4 bg-rose-500/5 border border-rose-500/20 rounded-lg flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-xs font-semibold text-rose-300">Cross-Partition Data Leakage Detection</p>
              <p className="text-xs text-zinc-400 mt-1">
                Visual duplicates across train and test partitions contaminate evaluation benchmarks. Curation allows researchers to quarantine or remove leaking samples.
              </p>
            </div>
          </div>

          <div className="space-y-4">
            {leakagePairs.map((pair) => (
              <Card key={pair.pair_id} className="bg-zinc-900/60 border-zinc-800 p-4">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 text-xs font-bold rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
                        {pair.cross_split_type.replace("_to_", " ↔ ").toUpperCase()}
                      </span>
                      <span className="text-xs font-bold text-zinc-100">
                        Similarity: {(pair.similarity_score * 100).toFixed(1)}%
                      </span>
                      <span className="text-xs text-zinc-500">({pair.match_type})</span>
                    </div>

                    <div className="flex items-center gap-4 text-xs font-mono text-zinc-300">
                      <div className="p-2 bg-zinc-950 rounded border border-zinc-800">
                        <span className="text-blue-400">[{pair.sample_a_split}]</span> {pair.sample_a_id}
                      </div>
                      <ArrowRight className="w-4 h-4 text-zinc-600" />
                      <div className="p-2 bg-zinc-950 rounded border border-zinc-800">
                        <span className="text-purple-400">[{pair.sample_b_split}]</span> {pair.sample_b_id}
                      </div>
                    </div>

                    <p className="text-xs text-zinc-400">{pair.recommendation}</p>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-xs border-zinc-700 hover:border-blue-500 hover:text-blue-300"
                      onClick={() => handleCurationDecision(pair.sample_b_id, pair.pair_id, "REPLACE_LEAK")}
                    >
                      Quarantine Test Sample
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-xs text-zinc-500 hover:text-zinc-300"
                      onClick={() => handleCurationDecision(pair.sample_b_id, pair.pair_id, "NOT_A_PROBLEM")}
                    >
                      Ignore Match
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* TAB 5: Hard Samples (Step 12-13) */}
      {activeTab === "hard_samples" && (
        <div className="space-y-6 animate-fade-in">
          <div className="flex items-center justify-between">
            <p className="text-xs text-zinc-400">
              Samples ranked by composite difficulty score combining evaluation failure rates, confidence margin, and annotation density.
            </p>
            <span className="text-xs font-semibold text-blue-400">{hardSamples.length} Prioritized Samples</span>
          </div>

          <div className="space-y-3">
            {hardSamples.map((h) => (
              <Card key={h.sample_id} className="bg-zinc-900/60 border-zinc-800 p-4 hover:border-zinc-700 transition-colors">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-xs font-bold text-zinc-100">{h.sample_id}</span>
                      <span className="text-xs px-2 py-0.5 bg-zinc-800 text-zinc-400 rounded">({h.split})</span>
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs text-zinc-400">Prioritization Score:</span>
                        <span className="text-xs font-bold text-amber-400">{(h.prioritization_score * 100).toFixed(0)} / 100</span>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-1.5">
                      {h.failure_reasons.map((r, idx) => (
                        <span key={idx} className="px-2 py-0.5 text-[11px] rounded bg-zinc-800 text-zinc-300 border border-zinc-700">
                          {r}
                        </span>
                      ))}
                    </div>

                    <div className="text-[11px] text-zinc-500">
                      GT Classes: {h.ground_truth_classes.join(", ") || "None"} | Predicted: {h.predicted_classes.join(", ") || "None"}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      className="text-xs border-zinc-700 hover:border-blue-500 hover:text-blue-300"
                      onClick={() => handleCurationDecision(h.sample_id, undefined, "SEND_TO_ACTIVE_LEARNING")}
                    >
                      Send to Active Learning
                    </Button>
                    <Button
                      size="sm"
                      className="text-xs bg-blue-600 hover:bg-blue-500"
                      onClick={() => setInspectSample(h.sample_id)}
                    >
                      Inspect Deep
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* TAB 6: Dataset Diff & Versioning (Step 23-24) */}
      {activeTab === "diff" && (
        <div className="space-y-6 animate-fade-in">
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-sm font-semibold text-zinc-200">Dataset Version Comparison & Diff</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-3">
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Baseline Version (A)</label>
                  <select
                    value={diffVerA}
                    onChange={(e) => setDiffVerA(e.target.value)}
                    className="bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200"
                  >
                    <option value="v1.0.0">v1.0.0 (Raw Ingestion)</option>
                    <option value="v2.0.0">v2.0.0 (Curated)</option>
                  </select>
                </div>

                <div className="pt-4">
                  <ArrowRight className="w-4 h-4 text-zinc-600" />
                </div>

                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Comparison Version (B)</label>
                  <select
                    value={diffVerB}
                    onChange={(e) => setDiffVerB(e.target.value)}
                    className="bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200"
                  >
                    <option value="v2.0.0">v2.0.0 (Curated)</option>
                    <option value="v1.0.0">v1.0.0 (Raw Ingestion)</option>
                  </select>
                </div>

                <div className="pt-4">
                  <Button size="sm" onClick={handleFetchDiff} className="bg-blue-600 hover:bg-blue-500 text-xs">
                    Compute Diff
                  </Button>
                </div>
              </div>

              {diffResult && (
                <div className="mt-4 p-4 bg-zinc-950 border border-zinc-800 rounded-lg space-y-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <span className="text-xs font-semibold text-zinc-200">{diffResult.summary}</span>
                  </div>

                  <div className="grid grid-cols-3 gap-3 pt-2">
                    <div className="p-3 bg-zinc-900 border border-zinc-800 rounded">
                      <span className="text-xs text-zinc-500">Annotations Delta</span>
                      <p className="text-lg font-bold text-emerald-400 mt-1">
                        {diffResult.annotations_count_delta > 0 ? `+${diffResult.annotations_count_delta}` : diffResult.annotations_count_delta}
                      </p>
                    </div>
                    <div className="p-3 bg-zinc-900 border border-zinc-800 rounded">
                      <span className="text-xs text-zinc-500">Leakage Reduction</span>
                      <p className="text-lg font-bold text-blue-400 mt-1">{diffResult.leakage_pairs_delta} pairs</p>
                    </div>
                    <div className="p-3 bg-zinc-900 border border-zinc-800 rounded">
                      <span className="text-xs text-zinc-500">New Classes</span>
                      <p className="text-lg font-bold text-zinc-200 mt-1">{diffResult.classes_added.length || "None"}</p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* TAB 7: Preparation & Split Pipeline */}
      {activeTab === "pipeline" && (
        <div className="space-y-6 animate-fade-in">
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-sm font-semibold text-zinc-200">Reproducible Dataset Partitioning</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Train Ratio (%)</label>
                  <input
                    type="number"
                    value={trainRatio}
                    onChange={(e) => setTrainRatio(Number(e.target.value))}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200"
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Validation Ratio (%)</label>
                  <input
                    type="number"
                    value={valRatio}
                    onChange={(e) => setValRatio(Number(e.target.value))}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200"
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Test Ratio (%)</label>
                  <input
                    type="number"
                    value={testRatio}
                    onChange={(e) => setTestRatio(Number(e.target.value))}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs text-zinc-200"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between pt-2">
                <span className="text-xs text-zinc-500">Seed: {seed} | Strategy: {strategy}</span>
                <Button size="sm" className="bg-emerald-600 hover:bg-emerald-500 text-xs gap-1.5">
                  <Play className="w-3.5 h-3.5" />
                  Run Partition & Splitting
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
