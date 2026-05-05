# Network exposure

PLAN.md §9. Default is **localhost only**. LAN exposure is a single opt-in
flag in `infra/serve.toml`.

## Layers

```
       browser ──► Caddy (LAN IP, :proxy_port, bearer-token guard)
                     │
                     ▼
                 picker (127.0.0.1:8088)
                     │
                     ▼
                 ollama (127.0.0.1:11434)
```

The picker and Ollama always bind to `127.0.0.1`. They are never directly
reachable from the LAN. The only thing on the LAN IP is Caddy.

## Switching to LAN

1. Edit `infra/serve.toml`:

   ```toml
   [network]
   expose = "lan"
   bind_addr = "auto"        # or a specific IP
   lan_cidr  = "auto"        # or e.g. "192.168.1.0/24"
   bearer_token_file = "/etc/chbt_nn/token"
   proxy_port = 8080
   ```

2. Re-run `./infra/bootstrap.sh`. It will:
   - Generate a 32-byte token at `/etc/chbt_nn/token` if missing.
   - Install Caddy if missing.
   - Print where to put `infra/Caddyfile.example` (after substituting
     `LAN_BIND`, `PROXY_PORT`, `TOKEN`).
   - Print the example nftables rule restricting `proxy_port` to the LAN.

3. From a phone/laptop on the LAN:

   ```bash
   curl -H "Authorization: Bearer $(cat /etc/chbt_nn/token)" \
        http://192.168.1.42:8080/api/health
   ```

## What we explicitly don't do

- **Public internet exposure.** Not a config option. PLAN.md §9 leaves
  further hardening (TLS, ACME, public DNS, fail2ban, etc.) to the
  maintainer.
- **Multi-user accounts.** PLAN.md §13.3: single user assumed.
- **Telemetry.** Nothing leaves the box.

## `lan-open` (no auth)

Available but not recommended. Use only on a fully trusted LAN segment.
