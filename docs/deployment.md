# VisionForge Deployment & Operations Guide

---

## 1. Quick Start Deployment

VisionForge is fully containerized and can be launched with a single command on macOS, Linux, or Windows (WSL2):

```bash
# 1. Clone the repository
git clone https://github.com/mzayan-bit/VisionForge.git
cd VisionForge

# 2. Copy and customize environment variables
cp .env.example .env

# 3. Start services with Docker Compose
docker compose up -d --build
# or using the task runner:
make up
```

Once running, access the services:
- **VisionForge Web UI**: [http://localhost:3000](http://localhost:3000)
- **Backend API & Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 2. Docker Compose Architecture

VisionForge's Compose configuration orchestrates two primary multi-stage containers interconnected via a private bridge network:

```yaml
services:
  backend:
    build: ./backend
    container_name: visionforge-backend
    ports: ["8000:8000"]
    volumes: [visionforge_data:/data]
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]

  frontend:
    build: ./frontend
    container_name: visionforge-frontend
    ports: ["3000:3000"]
    depends_on:
      backend:
        condition: service_healthy
```

### Key Deployment Characteristics:
1. **Multi-Stage Builds**: Strips compilers and build-time caches from production container layers.
2. **Non-Root Execution**: Backend runs as `appuser` (UID 10001) and Frontend runs as `nextjs` (UID 1001).
3. **Internal Reverse Proxy**: Frontend Next.js standalone server forwards all `/api/v1/*` requests internally to `http://backend:8000`, preventing browser CORS mismatches.
4. **Persistent Named Volume**: All checkpoints, datasets, and experiment records are preserved in `visionforge_data`.

---

## 3. Environment Variable Reference

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | **Required** | `production` | Active mode (`development`, `staging`, `production`, `testing`). |
| `DEBUG` | **Required** | `false` | Enables verbose stack traces (keep `false` in production). |
| `LOG_LEVEL` | **Required** | `INFO` | Logging severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `BACKEND_PORT` | **Required** | `8000` | Published host port for Backend API. |
| `FRONTEND_PORT` | **Required** | `3000` | Published host port for Frontend Web UI. |
| `BACKEND_INTERNAL_URL` | **Required** | `http://backend:8000` | Internal network URL used by Next.js API proxy rewrite. |
| `DATA_DIR` | **Required** | `/data` | Root path for persistent storage volume. |
| `MODEL_CACHE_DIR` | **Required** | `/data/models` | Storage path for trained model weights. |
| `DEFAULT_DEVICE` | **Required** | `auto` | Target compute engine (`auto`, `cpu`, `cuda`, `mps`). |
| `MAX_CACHED_MODELS` | **Required** | `3` | Maximum concurrent loaded models in memory cache. |
| `DATABASE_URL` | *Optional* | `None` | Optional PostgreSQL connection string. |
| `REDIS_URL` | *Optional* | `None` | Optional Redis broker URL. |
| `QDRANT_URL` | *Optional* | `None` | Optional Qdrant vector search server URL. |
| `NEO4J_URL` | *Optional* | `None` | Optional Neo4j graph database URL. |
| `MLFLOW_TRACKING_URI` | *Optional* | `None` | Optional remote MLflow server URI. |
| `OPENAI_API_KEY` | *Optional* | `None` | Optional API key for cloud vision-language models. |
| `ANTHROPIC_API_KEY` | *Optional* | `None` | Optional API key for cloud vision-language models. |

---

## 4. Operational Management Commands

VisionForge provides a centralized `Makefile` for day-to-day operations:

```bash
# View all available management targets
make help

# View live container logs
make logs

# Query backend health status
make health

# Seed the environment with real COCO8 dataset
make seed

# Run automated tests across the codebase
make test

# Gracefully stop all containers
make down

# [DESTRUCTIVE] Completely reset local development cache & volumes
make reset-dev
```

---

## 5. Storage Persistence & Backup Procedures

### Volume Structure (`/data` or `~/.cache/visionforge/`):
```text
/data/
├── datasets/         # Prepared dataset manifests & partitioned splits
├── models/           # Registered and installed model weights (*.pt)
├── memory/           # Visual Memory 768D vector index & records
├── training/         # Historical training runs and metrics snapshots
├── evaluations/      # Benchmark results, PR curves, and failure logs
├── explanations/     # Grad-CAM attribution heatmaps
└── experiments/      # Research experiments and lineage graphs
```

### Backup Volume:
```bash
# Create timestamped backup archive of visionforge_data volume
docker run --rm -v visionforge_data:/data -v $(pwd):/backup alpine \
  tar -czf /backup/visionforge_data_backup_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
```

### Restore Volume:
```bash
# Restore from backup archive
docker run --rm -v visionforge_data:/data -v $(pwd):/backup alpine \
  sh -c "rm -rf /data/* && tar -xzf /backup/<archive_name>.tar.gz -C /data"
```

---

## 6. Troubleshooting & Diagnostics

### Issue 1: Port `3000` or `8000` already in use
**Symptom**: `Error starting userland proxy: listen tcp 0.0.0.0:8000: bind: address already in use`
**Resolution**: Set alternative host ports in `.env`:
```bash
BACKEND_PORT=8080
FRONTEND_PORT=3030
docker compose up -d
```

### Issue 2: Backend Container Unhealthy
**Symptom**: `visionforge-backend (unhealthy)`
**Resolution**: Check container logs:
```bash
docker compose logs backend
# Inspect healthcheck probe
docker compose exec backend curl -i http://localhost:8000/health
```

### Issue 3: Frontend Cannot Reach Backend
**Symptom**: Web UI shows network error or loading spinner.
**Resolution**: Ensure `BACKEND_INTERNAL_URL` in `docker-compose.yml` is set to `http://backend:8000` and both containers reside on `visionforge_net`.

### Issue 4: Permissions on Persistent Volume
**Symptom**: `PermissionDenied` when saving checkpoints or datasets in Docker.
**Resolution**: The backend runs as user UID `10001`. Volume directories are automatically initialized with proper ownership. If mounting a host directory in development, ensure write access:
```bash
chmod -R 777 ./local_data_dir
```
