#!/bin/bash
set -euxo pipefail

cat >/etc/ecs/ecs.config <<'EOF'
ECS_CLUSTER=food-detection-v2-cluster
ECS_ENABLE_GPU_SUPPORT=true
ECS_AVAILABLE_LOGGING_DRIVERS=["json-file","awslogs"]
EOF

mkdir -p /data/v2/trellis_model /data/v2/trellis_dino /data/v2/trellis_rembg /data/v2/hf_cache

systemctl enable --now ecs
