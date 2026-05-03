#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CERTS_DIR="$SCRIPT_DIR/certs"
MKCERT="$HOME/.local/bin/mkcert"
MKCERT_VERSION="v1.4.4"
MKCERT_URL="https://github.com/FiloSottile/mkcert/releases/download/${MKCERT_VERSION}/mkcert-${MKCERT_VERSION}-linux-amd64"

# ---------- helpers ----------
command_exists() { command -v "$1" &>/dev/null; }

install_mkcert() {
  if command_exists mkcert; then
    MKCERT="mkcert"
    return
  fi
  if [[ -x "$MKCERT" ]]; then
    return
  fi
  echo "=== 下载 mkcert ==="
  mkdir -p "$(dirname "$MKCERT")"
  curl -fsSL "$MKCERT_URL" -o "$MKCERT"
  chmod +x "$MKCERT"
}

install_certutil() {
  if command_exists certutil; then
    return
  fi
  local tmp
  tmp="$(mktemp -d)"
  echo "=== 下载 libnss3-tools (certutil) ==="
  apt download libnss3-tools -o "Dir=${tmp}" 2>/dev/null || {
    rm -rf "$tmp"
    echo "[WARN] 无法下载 certutil，跳过 Chrome 信任安装"
    return 1
  }
  dpkg-deb -x "${tmp}/archives/libnss3-tools"*.deb "$HOME/.local/nss-tools"
  rm -rf "$tmp"
  export PATH="$HOME/.local/nss-tools/usr/bin:$PATH"
  export LD_LIBRARY_PATH="$HOME/.local/nss-tools/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

# ---------- main ----------
echo "=== deepseek2api HTTPS 证书一键安装 ==="

install_mkcert
echo "[1/3] mkcert 就绪: $MKCERT"

# Generate local CA + certs
export CAROOT="$HOME/.local/share/mkcert"
mkdir -p "$CERTS_DIR" "$CAROOT"

"$MKCERT" -cert-file "$CERTS_DIR/localhost.crt" \
          -key-file "$CERTS_DIR/localhost.key" \
          localhost 127.0.0.1 2>&1 | grep -v "^$"

echo "[2/3] 证书已生成: $CERTS_DIR/localhost.{crt,key}"

# Try to install into Chrome trust store
if install_certutil 2>/dev/null; then
  NSSDB="$HOME/.pki/nssdb"
  if [[ -f "$NSSDB/cert9.db" ]]; then
    certutil -A -d "sql:$NSSDB" -t "C,," -n "mkcert-localhost" \
      -i "$CAROOT/rootCA.pem" 2>/dev/null && \
      echo "[3/3] CA 已导入 Chrome 信任库" || \
      echo "[3/3] [WARN] 导入 Chrome 失败，请手动导入: $CAROOT/rootCA.pem"
  else
    echo "[3/3] [WARN] 未找到 Chrome NSS 数据库，请手动导入: $CAROOT/rootCA.pem"
  fi
else
  echo "[3/3] [WARN] 无法自动安装 Chrome 信任，请手动导入: $CAROOT/rootCA.pem"
fi

echo ""
echo "=== 完成 ==="
echo "证书: $CERTS_DIR/localhost.crt"
echo "密钥: $CERTS_DIR/localhost.key"
echo "CA 根证书: $CAROOT/rootCA.pem"
echo ""
echo "现在用 DEBUG=1 npm start 启动即可"
