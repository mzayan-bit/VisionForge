import React from "react";
import { BookOpen, Terminal, Layers, Shield } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card, CardBody } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";

export default function DocumentationPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Documentation & Specifications"
        description="Architecture design guides, API endpoints, design systems, and developer tooling."
        breadcrumbs={["VisionForge", "Documentation"]}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardBody className="space-y-3">
            <div className="flex items-center gap-2">
              <Terminal className="w-5 h-5 text-blue-400" />
              <h3 className="text-base font-semibold text-white font-geist">
                Backend Architecture
              </h3>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed">
              FastAPI ASGI service with Pydantic v2 Settings configuration, structured ANSI console logging, <code>APIResponse[T]</code> generic response envelopes, and centralized exception handling.
            </p>
            <div className="pt-2 flex items-center gap-2">
              <Badge variant="info" size="sm">FastAPI 0.111+</Badge>
              <Badge variant="neutral" size="sm">Python 3.11+</Badge>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardBody className="space-y-3">
            <div className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-emerald-400" />
              <h3 className="text-base font-semibold text-white font-geist">
                Frontend Design System
              </h3>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Built on Next.js 16 App Router using React 19 and Tailwind CSS v4. Designed with Stitch AI MCP applying an obsidian dark-mode first glassmorphic aesthetic inspired by Linear and Vercel.
            </p>
            <div className="pt-2 flex items-center gap-2">
              <Badge variant="success" size="sm">React 19</Badge>
              <Badge variant="info" size="sm">Next.js App Router</Badge>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardBody className="space-y-3">
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-purple-400" />
              <h3 className="text-base font-semibold text-white font-geist">
                API Organization & Tracing
              </h3>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Versioned API routers mounted at <code>/api/v1/</code> (`/health` and `/system/info`). Includes <code>RequestTracingMiddleware</code> attaching <code>X-Request-ID</code> and process timing headers.
            </p>
            <div className="pt-2 flex items-center gap-2">
              <Badge variant="neutral" size="sm">/api/v1/health</Badge>
              <Badge variant="neutral" size="sm">/api/v1/system/info</Badge>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardBody className="space-y-3">
            <div className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-amber-400" />
              <h3 className="text-base font-semibold text-white font-geist">
                Developer Quality Tools
              </h3>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Automated script <code>./scripts/lint.sh</code> executing Ruff linting/formatting, Pytest backend unit tests, TypeScript typechecking, and Next.js static build checks.
            </p>
            <div className="pt-2 flex items-center gap-2">
              <Badge variant="warning" size="sm">./scripts/lint.sh</Badge>
            </div>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
