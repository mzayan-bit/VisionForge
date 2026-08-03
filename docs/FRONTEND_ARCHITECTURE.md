# VisionForge Frontend Architecture Specification

## 🎨 Overview & Design Philosophy

VisionForge Workbench UI is built using **Next.js 16 (App Router)**, **React 19**, and **Tailwind CSS v4**.

Designed using **Stitch AI MCP**, the interface embodies a **Dark-Mode First**, **Obsidian Glassmorphic** aesthetic inspired by modern technical platforms such as Linear, Cursor, Vercel, and Arc Browser.

---

## 🏛️ 5-Pane Workbench Shell Layout

```text
+-----------------------------------------------------------------------------------+
| TopNav (Brand Logo, Global Search / Command Palette ⌘K, Environment & Quick Actions) |
+------------------+------------------------------------------+---------------------+
|                  |                                          |                     |
| Sidebar          | Main Workspace Canvas                    | Right Context Panel |
|                  |                                          |                     |
| - Home           | - Page Header & Breadcrumbs              | - Active Details    |
| - Workspace      | - Intentional Welcome Workspace          | - Runtime Metadata  |
| - Models         | - Hero Banner & Development Milestones   | - Health & Metrics  |
| - Benchmarks     | - Quick Action Operation Cards           |                     |
| - Experiments    | - Recent Activity Stream                 |                     |
| - Datasets       |                                          |                     |
| - Documentation  |                                          |                     |
| - Settings       |                                          |                     |
|                  |                                          |                     |
+------------------+------------------------------------------+---------------------+
| StatusBar (System Status, Active Route, Latency Metrics, Environment & Branch)   |
+-----------------------------------------------------------------------------------+
```

### 1. Top Navigation (`src/components/layout/TopNav.tsx`)
- Persistent header containing project branding (`VisionForge v0.1.0`), global command palette trigger (`⌘K`), quick documentation links, system status indicators, and active user badge.

### 2. Left Sidebar (`src/components/layout/Sidebar.tsx`)
- Collapsible navigation panel routing cleanly between core modules (`/`, `/workspace`, `/models`, `/benchmarks`, `/experiments`, `/datasets`, `/documentation`, `/settings`) with active indicator styling and badges.

### 3. Main Workspace Area (`src/app/page.tsx` & Page Components)
- Fluid viewport rendering page-specific content with standard `PageHeader`, `SectionHeader`, `Card` containers, and `EmptyState` components.

### 4. Right Context Inspector Panel (`src/components/layout/RightContextPanel.tsx`)
- Collapsible 280px inspection panel displaying tabbed contextual data (`Details`, `Runtime`, `Health`).

### 5. Bottom Status Bar (`src/components/layout/StatusBar.tsx`)
- Real-time status ticker bar providing operational indicators, current route path, latency telemetry (`24ms`), active branch (`main`), and environment mode (`DEVELOPMENT`).

---

## 💎 Design System & Tokens

### Color Palette (Obsidian Scale)
- **Base Canvas (`Level 0`)**: `#0A0A0A`
- **Surface Panels (`Level 1`)**: `#111111` with `rgba(255, 255, 255, 0.08)` borders (`border-white/10`)
- **Modals / Popovers (`Level 2`)**: `#171717` (80% opacity with `backdrop-blur-md` glassmorphism)
- **Primary Accent**: Electric Blue (`#2E90FA` / `blue-600`)
- **Secondary Accent**: Emerald Green (`#12B76A` / `emerald-400`)
- **Tertiary Accent**: Amber (`#F79009` / `amber-400`)

### Typography
- **Headings & UI Controls**: `Geist` font family with tight letter-spacing.
- **Body Content**: `Inter` font family for high-density readability.
- **Monospaced Data & Code**: Monospaced font for route paths, latency telemetry, and model parameters.

### Spacing & Elevation
- **Grid Rhythm**: 4px base linear unit (8px/16px/24px/32px spacing scale).
- **Corner Radii**: Soft UI corners (`0.25rem` / `4px` for inputs/badges, `0.5rem` / `8px` for containers, `0.75rem` / `12px` for hero cards).

---

## 🧩 Reusable Component Library (`src/components/ui/`)

| Component | File Path | Purpose |
| :--- | :--- | :--- |
| **`Button`** | [src/components/ui/Button.tsx](file:///Users/zayan/Documents/VisionForge/frontend/src/components/ui/Button.tsx) | Primary, Secondary, Outline, Ghost, Danger button variants |
| **`Card`** | [src/components/ui/Card.tsx](file:///Users/zayan/Documents/VisionForge/frontend/src/components/ui/Card.tsx) | `Card`, `CardHeader`, `CardBody`, `CardFooter` containers |
| **`Badge`** | [src/components/ui/Badge.tsx](file:///Users/zayan/Documents/VisionForge/frontend/src/components/ui/Badge.tsx) | Status indicators (Success, Warning, Error, Info, Neutral) with optional live dot |
| **`PageHeader`** | [src/components/ui/PageHeader.tsx](file:///Users/zayan/Documents/VisionForge/frontend/src/components/ui/PageHeader.tsx) | Page titles, descriptions, breadcrumbs navigation, and action slots |
| **`SectionHeader`** | [src/components/ui/SectionHeader.tsx](file:///Users/zayan/Documents/VisionForge/frontend/src/components/ui/SectionHeader.tsx) | Subsection title headers with tracking, count badges, and action links |
| **`Modal`** | [src/components/ui/Modal.tsx](file:///Users/zayan/Documents/VisionForge/frontend/src/components/ui/Modal.tsx) | Accessible dialog backdrop overlay with Escape key handler |
| **`CommandPalette`** | [src/components/ui/CommandPalette.tsx](file:///Users/zayan/Documents/VisionForge/frontend/src/components/ui/CommandPalette.tsx) | Floating search & command trigger modal (`⌘K`) |
| **`EmptyState`** | [src/components/ui/EmptyState.tsx](file:///Users/zayan/Documents/VisionForge/frontend/src/components/ui/EmptyState.tsx) | Reusable intentional empty workspace placeholder |
| **`LoadingState`** | [src/components/ui/LoadingState.tsx](file:///Users/zayan/Documents/VisionForge/frontend/src/components/ui/LoadingState.tsx) | Spinner indicators and `Skeleton` pulse loaders |
| **`Tabs`** | [src/components/ui/Tabs.tsx](file:///Users/zayan/Documents/VisionForge/frontend/src/components/ui/Tabs.tsx) | Tab list navigation with badges and icons |

---

## ⚡ Performance & Accessibility

1. **Server vs. Client Component Boundaries**: Top-level pages remain React Server Components where possible; `"use client"` is isolated to interactive controls (Command Palette, Sidebar collapse state, Modal toggles).
2. **Keyboard Accessibility**: Full keyboard support for Command Palette (`⌘K` to open, `ESC` to dismiss), clear focus states (`focus:ring-2 focus:ring-blue-500/50`), and semantic HTML tags (`<header>`, `<nav>`, `<aside>`, `<main>`, `<footer>`).
3. **Optimized Production Build**: Zero ESLint or TypeScript compilation errors; fully static page prerendering in Next.js.
