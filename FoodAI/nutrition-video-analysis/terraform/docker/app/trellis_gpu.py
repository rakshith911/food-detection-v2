"""
TRELLIS GPU task manager for v2 pipeline.

Handles EC2 GPU instance lifecycle (start/stop) and ECS task submission.
One g6.2xlarge GPU instance, one task at a time; instance is stopped after
each job to save cost.
"""
import logging
import time
from pathlib import Path
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

_STOP_WAIT_S = 300   # 5 min wait after issuing stop
_START_WAIT_S = 300  # 5 min wait before launching ECS task after start


def _ec2(region: str):
    return boto3.client("ec2", region_name=region)


def _ecs(region: str):
    return boto3.client("ecs", region_name=region)


def _s3(region: str):
    return boto3.client("s3", region_name=region)


# ── Instance lifecycle ─────────────────────────────────────────────────────────

def _instance_state(instance_id: str, region: str) -> str:
    resp = _ec2(region).describe_instances(InstanceIds=[instance_id])
    return resp["Reservations"][0]["Instances"][0]["State"]["Name"]


def ensure_instance_running(instance_id: str, region: str, job_id: str) -> None:
    state = _instance_state(instance_id, region)
    logger.info("[%s] GPU instance %s state: %s", job_id, instance_id, state)

    if state == "running":
        return

    if state in ("stopping", "stopped"):
        if state == "stopping":
            logger.info("[%s] Instance stopping — waiting %ds for it to fully stop", job_id, _STOP_WAIT_S)
            time.sleep(_STOP_WAIT_S)

        logger.info("[%s] Starting GPU instance %s", job_id, instance_id)
        _ec2(region).start_instances(InstanceIds=[instance_id])

        # Poll until running (up to 10 min)
        for _ in range(60):
            time.sleep(10)
            s = _instance_state(instance_id, region)
            logger.info("[%s] Instance state: %s", job_id, s)
            if s == "running":
                break
        else:
            raise RuntimeError(f"[{job_id}] GPU instance {instance_id} did not reach 'running' in time")

        logger.info("[%s] Waiting %ds for ECS agent to register", job_id, _START_WAIT_S)
        time.sleep(_START_WAIT_S)
        return

    raise RuntimeError(f"[{job_id}] GPU instance in unexpected state: {state}")


def stop_instance(instance_id: str, region: str, job_id: str) -> None:
    try:
        state = _instance_state(instance_id, region)
        if state not in ("running", "pending"):
            logger.info("[%s] Instance already %s — skip stop", job_id, state)
            return
        logger.info("[%s] Stopping GPU instance %s", job_id, instance_id)
        _ec2(region).stop_instances(InstanceIds=[instance_id])
        # Don't wait — caller can fire-and-forget; next job waits as needed
    except Exception as e:
        logger.warning("[%s] Could not stop GPU instance: %s", job_id, e)


# ── ECS task submission ────────────────────────────────────────────────────────

def _container_instance_arn(cluster: str, region: str) -> Optional[str]:
    resp = _ecs(region).list_container_instances(cluster=cluster, status="ACTIVE")
    arns = resp.get("containerInstanceArns") or []
    return arns[0] if arns else None


def submit_trellis_task(
    *,
    cluster: str,
    task_definition: str,
    input_bucket: str,
    input_keys: list[str],
    output_bucket: str,
    output_prefix: str,
    preview_seconds: int,
    preview_fps: int,
    region: str,
    job_id: str,
) -> str:
    ci_arn = _container_instance_arn(cluster, region)
    if not ci_arn:
        raise RuntimeError(f"[{job_id}] No active container instance in cluster {cluster}")

    env_overrides = [
        {"name": "INPUT_BUCKET",     "value": input_bucket},
        {"name": "INPUT_KEYS",       "value": ",".join(input_keys)},
        {"name": "RESULTS_BUCKET",   "value": output_bucket},
        {"name": "OUTPUT_PREFIX",    "value": output_prefix},
        {"name": "PREVIEW_SECONDS",  "value": str(preview_seconds)},
        {"name": "PREVIEW_FPS",      "value": str(preview_fps)},
    ]

    resp = _ecs(region).start_task(
        cluster=cluster,
        taskDefinition=task_definition,
        containerInstances=[ci_arn],
        overrides={
            "containerOverrides": [{
                "name": "trellis-gpu",
                "environment": env_overrides,
            }]
        },
    )
    failures = resp.get("failures") or []
    if failures:
        raise RuntimeError(f"[{job_id}] ECS start_task failures: {failures}")

    task_arn = resp["tasks"][0]["taskArn"]
    logger.info("[%s] TRELLIS ECS task submitted: %s", job_id, task_arn)
    return task_arn


def wait_for_task(task_arn: str, cluster: str, region: str, job_id: str, timeout_s: int = 1800) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(30)
        resp = _ecs(region).describe_tasks(cluster=cluster, tasks=[task_arn])
        tasks = resp.get("tasks") or []
        if not tasks:
            raise RuntimeError(f"[{job_id}] Task {task_arn} not found in describe_tasks")
        task = tasks[0]
        status = task.get("lastStatus", "UNKNOWN")
        logger.info("[%s] ECS task status: %s", job_id, status)
        if status == "STOPPED":
            exit_code = None
            for c in task.get("containers") or []:
                exit_code = c.get("exitCode")
            if exit_code is not None and exit_code != 0:
                reason = task.get("stoppedReason", "unknown")
                raise RuntimeError(f"[{job_id}] TRELLIS task exited {exit_code}: {reason}")
            return status
    raise TimeoutError(f"[{job_id}] TRELLIS task did not complete within {timeout_s}s")


# ── S3 output helpers ──────────────────────────────────────────────────────────

def _s3_key_exists(bucket: str, key: str, region: str) -> bool:
    try:
        _s3(region).head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def upload_image_for_trellis(
    image_path: Path,
    bucket: str,
    prefix: str,
    region: str,
    job_id: str,
) -> str:
    key = f"{prefix.rstrip('/')}/{image_path.name}"
    _s3(region).upload_file(str(image_path), bucket, key)
    logger.info("[%s] Uploaded input image to s3://%s/%s", job_id, bucket, key)
    return key


def download_trellis_output(
    bucket: str,
    output_prefix: str,
    stem: str,
    region: str,
    job_id: str,
    local_dir: Path,
) -> dict:
    """Download GLB and MP4 for a dish stem. Returns dict with 'glb' and 'mp4' local paths."""
    out = {}
    for ext, content_type in [("glb", "model/gltf-binary"), ("mp4", "video/mp4")]:
        key = f"{output_prefix.rstrip('/')}/{stem}.{ext}"
        if _s3_key_exists(bucket, key, region):
            dest = local_dir / f"{stem}.{ext}"
            _s3(region).download_file(bucket, key, str(dest))
            out[ext] = dest
            logger.info("[%s] Downloaded s3://%s/%s → %s", job_id, bucket, key, dest)
        else:
            logger.warning("[%s] s3://%s/%s not found", job_id, bucket, key)
    return out


# ── High-level orchestrator ────────────────────────────────────────────────────

def run_trellis_for_job(
    *,
    image_paths: list[Path],
    config,
    job_id: str,
    local_output_dir: Path,
) -> dict:
    """
    Start GPU → upload images → submit ECS task → wait → download outputs → stop GPU.

    Returns: {stem: {"glb": Path, "mp4": Path, "glb_s3_key": str}} per image.
    """
    region = config.TRELLIS_AWS_REGION
    instance_id = config.TRELLIS_GPU_INSTANCE_ID
    cluster = config.TRELLIS_ECS_CLUSTER
    task_def = config.TRELLIS_TASK_DEFINITION
    in_bucket = config.TRELLIS_INPUT_BUCKET
    out_bucket = config.TRELLIS_OUTPUT_BUCKET
    in_prefix = config.TRELLIS_INPUT_PREFIX
    out_prefix = f"{config.TRELLIS_OUTPUT_PREFIX}/{job_id}"

    local_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ensure_instance_running(instance_id, region, job_id)

        # Upload input images to S3
        input_keys = [
            upload_image_for_trellis(p, in_bucket, f"{in_prefix}/{job_id}", region, job_id)
            for p in image_paths
        ]

        task_arn = submit_trellis_task(
            cluster=cluster,
            task_definition=task_def,
            input_bucket=in_bucket,
            input_keys=input_keys,
            output_bucket=out_bucket,
            output_prefix=out_prefix,
            preview_seconds=config.TRELLIS_PREVIEW_SECONDS,
            preview_fps=config.TRELLIS_PREVIEW_FPS,
            region=region,
            job_id=job_id,
        )

        wait_for_task(task_arn, cluster, region, job_id, timeout_s=config.TRELLIS_TASK_TIMEOUT_S)

    finally:
        stop_instance(instance_id, region, job_id)

    # Download outputs
    results = {}

    # Parse manifest.json for volume metrics
    manifest_by_stem = {}
    manifest_key = f"{out_prefix}/manifest.json"
    try:
        import json as _json, tempfile as _tmp
        _mf = Path(_tmp.mktemp(suffix=".json"))
        _s3(region).download_file(out_bucket, manifest_key, str(_mf))
        for entry in _json.loads(_mf.read_text()):
            s = Path(entry.get("input_image", "")).stem
            if s:
                manifest_by_stem[s] = entry
        _mf.unlink(missing_ok=True)
        logger.info("[%s] Parsed TRELLIS manifest (%d entries)", job_id, len(manifest_by_stem))
    except Exception as _me:
        logger.warning("[%s] Could not parse TRELLIS manifest: %s", job_id, _me)

    for img_path in image_paths:
        stem = img_path.stem
        files = download_trellis_output(out_bucket, out_prefix, stem, region, job_id, local_output_dir)
        glb_key = f"{out_prefix}/{stem}.glb"
        entry = manifest_by_stem.get(stem, {})
        results[stem] = {
            **files,
            "glb_s3_key": glb_key if "glb" in files else None,
            "food_volume_units": entry.get("food_volume_units"),
            "vessel_diameter_units": entry.get("vessel_diameter_units"),
            "mesh_metadata": entry.get("mesh_metadata") or {},
        }

    return results
