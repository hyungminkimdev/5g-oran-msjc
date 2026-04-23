#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Open5GS 5G Core Setup — MSJC Project (Instance-2)
# Installs Open5GS, configures AMF/UPF, registers test subscriber
# Target: Ubuntu 22.04 LTS
# ─────────────────────────────────────────────────────────────
set -e

PLMN_MCC="001"
PLMN_MNC="01"
TAC="7"
AMF_NGAP_ADDR="127.0.0.5"
UPF_GTPU_ADDR="127.0.0.7"
UPF_TUN="ogstun"
UPF_TUN_ADDR="10.45.0.1/16"

# Test subscriber credentials (must match srsUE ue_msjc.conf)
IMSI="001010123456780"
KI="00112233445566778899AABBCCDDEEFF"
OPC="63BFA50EE6523365FF14C1F45F88737D"
APN="internet"

echo "============================================"
echo "[1/6] Installing dependencies"
echo "============================================"
sudo apt-get update
sudo apt-get install -y software-properties-common gnupg curl

echo "============================================"
echo "[2/6] Adding Open5GS repository and installing"
echo "============================================"
sudo add-apt-repository -y ppa:open5gs/latest
sudo apt-get update
sudo apt-get install -y open5gs

echo "============================================"
echo "[3/6] Configuring AMF — PLMN ${PLMN_MCC}${PLMN_MNC}, TAC ${TAC}"
echo "============================================"
AMF_CONF="/etc/open5gs/amf.yaml"
sudo cp "${AMF_CONF}" "${AMF_CONF}.bak"

# Set NGAP bind address
sudo sed -i "s|addr: 127.0.0.5|addr: ${AMF_NGAP_ADDR}|g" "${AMF_CONF}"

# Set PLMN
sudo python3 - "${AMF_CONF}" "${PLMN_MCC}" "${PLMN_MNC}" "${TAC}" <<'PYEOF'
import sys, yaml

conf_path, mcc, mnc, tac = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(conf_path) as f:
    cfg = yaml.safe_load(f)

# Navigate to amf.guami and amf.plmn_support
amf = cfg.get("amf", cfg)

# Update GUAMI
for guami in amf.get("guami", []):
    plmn_id = guami.get("plmn_id", {})
    plmn_id["mcc"] = mcc
    plmn_id["mnc"] = mnc

# Update PLMN support
for ps in amf.get("plmn_support", []):
    plmn_id = ps.get("plmn_id", {})
    plmn_id["mcc"] = mcc
    plmn_id["mnc"] = mnc
    # Update TAC
    ps["tac"] = int(tac)

# Update TAI
for tai in amf.get("tai", []):
    plmn_id = tai.get("plmn_id", {})
    plmn_id["mcc"] = mcc
    plmn_id["mnc"] = mnc
    tai["tac"] = int(tac)

with open(conf_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"  AMF configured: MCC={mcc} MNC={mnc} TAC={tac}")
PYEOF

echo "============================================"
echo "[4/6] Configuring UPF — GTP-U ${UPF_GTPU_ADDR}, TUN ${UPF_TUN}"
echo "============================================"
UPF_CONF="/etc/open5gs/upf.yaml"
sudo cp "${UPF_CONF}" "${UPF_CONF}.bak"

sudo sed -i "s|addr: 127.0.0.7|addr: ${UPF_GTPU_ADDR}|g" "${UPF_CONF}"

# Create TUN interface for user-plane
sudo ip tuntap add name ${UPF_TUN} mode tun 2>/dev/null || true
sudo ip addr add ${UPF_TUN_ADDR} dev ${UPF_TUN} 2>/dev/null || true
sudo ip link set ${UPF_TUN} up
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -s 10.45.0.0/16 ! -o ${UPF_TUN} -j MASQUERADE 2>/dev/null || true
echo "  UPF TUN interface ${UPF_TUN} up at ${UPF_TUN_ADDR}"

echo "============================================"
echo "[5/6] Registering test subscriber"
echo "============================================"
echo "  IMSI: ${IMSI}"
echo "  K:    ${KI}"
echo "  OPC:  ${OPC}"
echo "  APN:  ${APN}"

# open5gs-dbctl is the CLI tool to manage the MongoDB subscriber DB
if command -v open5gs-dbctl &>/dev/null; then
    open5gs-dbctl add "${IMSI}" "${KI}" "${OPC}" 2>/dev/null || true
    open5gs-dbctl type "${IMSI}" 1 2>/dev/null || true
    echo "  Subscriber registered via open5gs-dbctl"
else
    # Fallback: direct MongoDB insertion
    echo "  open5gs-dbctl not found, attempting direct MongoDB insert..."
    mongosh --quiet --eval "
        db = db.getSiblingDB('open5gs');
        db.subscribers.updateOne(
            { imsi: '${IMSI}' },
            {
                \$set: {
                    imsi: '${IMSI}',
                    security: {
                        k: '${KI}',
                        amf: '8000',
                        op_type: 2,
                        op_value: '${OPC}'
                    },
                    schema_version: 1,
                    slice: [{
                        sst: 1,
                        default_indicator: true,
                        session: [{
                            name: '${APN}',
                            type: 3,
                            qos: { index: 9, arp: { priority_level: 8, pre_emption_capability: 1, pre_emption_vulnerability: 1 } },
                            ambr: { downlink: { value: 1, unit: 3 }, uplink: { value: 1, unit: 3 } }
                        }]
                    }]
                }
            },
            { upsert: true }
        );
        print('  Subscriber registered via MongoDB');
    " 2>/dev/null || echo "  [WARN] MongoDB insert failed — register subscriber manually via Open5GS WebUI (http://localhost:9999)"
fi

echo "============================================"
echo "[6/6] Enabling and starting Open5GS services"
echo "============================================"
SERVICES=(
    open5gs-nrfd
    open5gs-scpd
    open5gs-amfd
    open5gs-smfd
    open5gs-upfd
    open5gs-ausfd
    open5gs-udmd
    open5gs-udrd
    open5gs-pcfd
    open5gs-nssfd
    open5gs-bsfd
)

for svc in "${SERVICES[@]}"; do
    sudo systemctl enable "${svc}" 2>/dev/null || true
    sudo systemctl restart "${svc}"
    echo "  ${svc}: $(sudo systemctl is-active ${svc})"
done

echo ""
echo "============================================"
echo "[DONE] Open5GS 5G Core is running"
echo "  AMF NGAP: ${AMF_NGAP_ADDR}:38412"
echo "  UPF GTPU: ${UPF_GTPU_ADDR}:2152"
echo "  WebUI:    http://localhost:9999 (admin/1423)"
echo "  Subscriber IMSI: ${IMSI}"
echo "============================================"
