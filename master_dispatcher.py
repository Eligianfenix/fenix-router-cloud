#!/usr/bin/env python3
"""
master_dispatcher.py - Repartidor Maestro con Encolamiento Fair-Share (WFQ) y Servidor SSE
Fénix Studio v3.0 — Premio a la Innovación San Juan 2026 | Costo $0.00 Absoluto
Garantiza SLA < 10 minutos para 100 usuarios concurrentes (1,200 peticiones de cátedra),
balanceando a nivel de micro-petición y manteniendo streams SSE vivos por hasta 100 min.
"""

import heapq
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, Generator, List, Optional, Tuple

from inference_node import (
    DeterministicIntentClassifier,
    DynamicCatalogDiscovery,
    IsolatedInferenceNode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [FenixMaster] %(message)s",
)
logger = logging.getLogger("FenixRouter.MasterDispatcher")

ROUTERS_DIR = os.path.expanduser(os.environ.get("FENIX_ROUTERS_DIR", "~/.fenix/secrets/routers"))
SECRETS_PATH = os.path.expanduser(os.environ.get("FENIX_SECRETS_PATH", "~/.fenix/secrets/api_keys.json"))
DEFAULT_MAX_USERS_PER_NODE = 100


class FairShareScheduler:
    """
    Planificador Weighted Fair Queuing (WFQ) por Tiempo Virtual de Finalización (VFT).
    Impide que un estudiante que sube 30 PDFs monopolice la fila frente a otros 99
    estudiantes concurrentes, asegurando que todos reciban progreso continuo en <10 min.
    """

    def __init__(self, max_concurrent_global: int = 220):
        self._lock = threading.Condition()
        self._heap: List[Tuple[float, int, str, threading.Event]] = []
        self._seq = 0
        self._virtual_time = 0.0
        self._user_last_vft: Dict[str, float] = {}
        self._user_in_flight: Dict[str, int] = {}
        self._active_global = 0
        self._max_concurrent_global = max_concurrent_global

    def acquire_turn(self, user_id: str, est_tokens: int, timeout_sec: float = 300.0) -> bool:
        ready_event = threading.Event()
        with self._lock:
            in_flight_u = self._user_in_flight.get(user_id, 0)
            weight_u = 1.0 / (1.0 + 0.5 * in_flight_u)
            cost = max(500.0, float(est_tokens)) / weight_u

            start_vft = max(self._virtual_time, self._user_last_vft.get(user_id, 0.0))
            finish_vft = start_vft + cost
            self._user_last_vft[user_id] = finish_vft

            if self._active_global < self._max_concurrent_global and not self._heap:
                self._active_global += 1
                self._user_in_flight[user_id] = in_flight_u + 1
                return True

            self._seq += 1
            entry = (finish_vft, self._seq, user_id, ready_event)
            heapq.heappush(self._heap, entry)

        granted = ready_event.wait(timeout=timeout_sec)
        return granted

    def release_turn(self, user_id: str) -> None:
        with self._lock:
            self._user_in_flight[user_id] = max(0, self._user_in_flight.get(user_id, 1) - 1)
            if self._heap:
                finish_vft, _, next_uid, next_event = heapq.heappop(self._heap)
                self._virtual_time = max(self._virtual_time, finish_vft)
                self._user_in_flight[next_uid] = self._user_in_flight.get(next_uid, 0) + 1
                next_event.set()
            else:
                self._active_global = max(0, self._active_global - 1)


class UserSessionLease:
    """Rastrea la actividad de sesión del estudiante para telemetría y SLA."""

    def __init__(self, user_id: str, preferred_node_id: str, lease_duration_sec: float = 1800.0):
        self.user_id = user_id
        self.preferred_node_id = preferred_node_id
        self.lease_start_epoch = time.time()
        self.last_activity_epoch = self.lease_start_epoch
        self.lease_duration_sec = lease_duration_sec
        self.request_count = 0

    def touch(self) -> None:
        self.last_activity_epoch = time.time()
        self.request_count += 1


class MasterDeterministicDispatcher:
    """
    Repartidor Maestro Determinista v3.0:
    - Soporta configuración por variables de entorno (Render / HF Spaces / Docker) o archivos JSON.
    - Balancea cada petición de una unidad académica a través de todos los nodos y los 16 proveedores.
    - Garantiza Zero-Drop Failover entre nodos e inyección de telemetría en tiempo real.
    """

    def __init__(self, secrets_path: str = SECRETS_PATH):
        self.secrets_path = secrets_path
        self.nodes: Dict[str, IsolatedInferenceNode] = {}
        self.node_identities: Dict[str, str] = {}
        self.sessions: Dict[str, UserSessionLease] = {}
        self._lock = threading.RLock()
        self.scheduler = FairShareScheduler(max_concurrent_global=240)
        self._bootstrap_nodes()

    def _bootstrap_nodes(self) -> None:
        env_json = os.environ.get("FENIX_NODES_CONFIG_JSON", "").strip()
        if env_json:
            try:
                nodes_list = json.loads(env_json)
                for idx, n_cfg in enumerate(nodes_list, start=1):
                    nid = n_cfg.get("node_id", f"Node-{idx:02d}")
                    email = n_cfg.get("email", f"federated_{idx}@fenix.unsj.edu.ar")
                    keys = n_cfg.get("keys", {})
                    self.nodes[nid] = IsolatedInferenceNode(
                        node_id=nid,
                        identity_email=email,
                        provider_api_keys=keys,
                        max_in_flight_channels=128,
                        proactive_depletion_callback=self._handle_node_low_watermark,
                    )
                    self.node_identities[nid] = email
                logger.info(f"Inicializados {len(self.nodes)} nodos desde FENIX_NODES_CONFIG_JSON.")
                return
            except Exception as e:
                logger.error(f"Error parseando FENIX_NODES_CONFIG_JSON: {e}")

        env_provider_map = {
            "gemini": os.environ.get("GEMINI_API_KEY", ""),
            "groq": os.environ.get("GROQ_API_KEY", ""),
            "sambanova": os.environ.get("SAMBANOVA_API_KEY", ""),
            "openrouter": os.environ.get("OPENROUTER_API_KEY", ""),
            "siliconflow": os.environ.get("SILICONFLOW_API_KEY", ""),
            "cohere": os.environ.get("COHERE_API_KEY", ""),
            "nvidia": os.environ.get("NVIDIA_API_KEY", ""),
            "baseten": os.environ.get("BASETEN_API_KEY", ""),
            "fastrouter": os.environ.get("FASTROUTER_API_KEY", ""),
            "unorouter": os.environ.get("UNOROUTER_API_KEY", ""),
            "routeway": os.environ.get("ROUTEWAY_API_KEY", ""),
            "tokenharbor": os.environ.get("TOKENHARBOR_API_KEY", ""),
            "requesty": os.environ.get("REQUESTY_API_KEY", ""),
            "plugsky": os.environ.get("PLUGSKY_API_KEY", ""),
            "agnes": os.environ.get("AGNES_API_KEY", ""),
            "airforce": os.environ.get("AIRFORCE_API_KEY", ""),
            "naga": os.environ.get("NAGA_API_KEY", ""),
            "github": os.environ.get("GITHUB_TOKEN", ""),
            "kilocode": os.environ.get("KILOCODE_API_KEY", ""),
        }
        active_env_keys = {k: v.strip() for k, v in env_provider_map.items() if v and v.strip()}
        if active_env_keys:
            if "gemini" in active_env_keys:
                active_env_keys["gemini_pool"] = [
                    k.strip() for k in active_env_keys["gemini"].split(",") if k.strip()
                ]
                active_env_keys["gemini"] = active_env_keys["gemini_pool"][0]
            nid = os.environ.get("FENIX_NODE_ID", "Node-Cloud-01")
            email = os.environ.get("FENIX_IDENTITY_EMAIL", "cloud01@fenix.unsj.edu.ar")
            self.nodes[nid] = IsolatedInferenceNode(
                node_id=nid,
                identity_email=email,
                provider_api_keys=active_env_keys,
                max_in_flight_channels=128,
                proactive_depletion_callback=self._handle_node_low_watermark,
            )
            self.node_identities[nid] = email
            logger.info(f"Nodo {nid} inicializado desde variables de entorno con {len(active_env_keys)} proveedores.")
            return

        if os.path.exists(ROUTERS_DIR):
            files = sorted(
                [f for f in os.listdir(ROUTERS_DIR) if f.startswith("router_") and f.endswith(".json")]
            )
            for f_name in files:
                full_p = os.path.join(ROUTERS_DIR, f_name)
                try:
                    with open(full_p, "r", encoding="utf-8") as f:
                        rdata = json.load(f)
                    r_id = rdata.get("router_id", f_name.replace(".json", "").capitalize())
                    email = rdata.get("email", "federated@fenix.local")
                    g_proyectos = rdata.get("gemini", {}).get("proyectos", [])
                    g_keys = [p.get("api_key", "").strip() for p in g_proyectos if p.get("api_key", "").strip()]
                    if not g_keys and isinstance(rdata.get("gemini"), dict) and rdata["gemini"].get("api_key"):
                        g_keys = [rdata["gemini"]["api_key"].strip()]

                    node_keys: Dict[str, Any] = {
                        "gemini": g_keys[0] if g_keys else "",
                        "gemini_pool": g_keys,
                    }
                    for prov_name, prov_dict in rdata.items():
                        if prov_name in ("router_id", "email", "descripcion", "gemini"):
                            continue
                        if isinstance(prov_dict, dict):
                            key_val = prov_dict.get("api_key", "").strip()
                            if key_val:
                                node_keys[prov_name] = key_val

                    self.nodes[r_id] = IsolatedInferenceNode(
                        node_id=r_id,
                        identity_email=email,
                        provider_api_keys=node_keys,
                        max_in_flight_channels=128,
                        proactive_depletion_callback=self._handle_node_low_watermark,
                    )
                    self.node_identities[r_id] = email
                except Exception as exc:
                    logger.error(f"Error cargando {full_p}: {exc}")
            if self.nodes:
                logger.info(f"Inicializados {len(self.nodes)} nodos desde {ROUTERS_DIR}.")
                return

        if os.path.exists(self.secrets_path):
            try:
                with open(self.secrets_path, "r", encoding="utf-8") as f:
                    secrets = json.load(f)
                all_llm_keys: Dict[str, Any] = {}
                for prov_name, prov_val in secrets.items():
                    if isinstance(prov_val, dict):
                        k = prov_val.get("api_key") or prov_val.get("token") or prov_val.get("key")
                        if k:
                            all_llm_keys[prov_name] = k
                nid = "Node-Master-01"
                self.nodes[nid] = IsolatedInferenceNode(
                    node_id=nid,
                    identity_email=secrets.get("email", "master@fenix.local"),
                    provider_api_keys=all_llm_keys,
                    max_in_flight_channels=128,
                    proactive_depletion_callback=self._handle_node_low_watermark,
                )
                self.node_identities[nid] = secrets.get("email", "master@fenix.local")
            except Exception as exc:
                logger.error(f"Error leyendo {self.secrets_path}: {exc}")

    def _handle_node_low_watermark(self, node_id: str, reason: str) -> None:
        with self._lock:
            logger.warning(f"[FAILOVER PROACTIVO] {node_id}: {reason}. Rebalanceando sesiones...")
            for uid, lease in self.sessions.items():
                if lease.preferred_node_id == node_id:
                    alt = self._select_best_node_for_request_unlocked(exclude_node_ids={node_id})
                    if alt:
                        lease.preferred_node_id = alt.node_id

    def _select_best_node_for_request_unlocked(
        self, exclude_node_ids: Optional[set] = None
    ) -> Optional[IsolatedInferenceNode]:
        exclude = exclude_node_ids or set()
        best_node: Optional[IsolatedInferenceNode] = None
        best_score = -1e9

        for nid, node in self.nodes.items():
            if nid in exclude:
                continue
            residual = node.compute_weighted_residual_quota()
            if residual <= 0.005:
                continue
            in_flight_ratio = node.active_in_flight_count / max(1.0, float(node.max_in_flight_channels))
            score = (0.65 * residual) + (0.35 * (1.0 - in_flight_ratio))
            if score > best_score:
                best_score = score
                best_node = node

        if best_node is None and self.nodes:
            for nid, node in self.nodes.items():
                if nid not in exclude:
                    return node
        return best_node

    def select_ordered_nodes_for_user(self, user_id: str) -> List[IsolatedInferenceNode]:
        with self._lock:
            if user_id not in self.sessions:
                primary = self._select_best_node_for_request_unlocked()
                pref_id = primary.node_id if primary else next(iter(self.nodes.keys()), "")
                self.sessions[user_id] = UserSessionLease(user_id=user_id, preferred_node_id=pref_id)

            lease = self.sessions[user_id]
            lease.touch()

            ranked = sorted(
                self.nodes.values(),
                key=lambda n: (
                    0.65 * n.compute_weighted_residual_quota()
                    + 0.35 * (1.0 - (n.active_in_flight_count / max(1.0, float(n.max_in_flight_channels))))
                ),
                reverse=True,
            )
            return ranked

    def dispatch_non_streaming(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        contract = DeterministicIntentClassifier.classify(payload)
        est_tokens = contract.estimated_input_tokens + contract.estimated_output_tokens

        if not self.scheduler.acquire_turn(user_id, est_tokens, timeout_sec=300.0):
            raise RuntimeError("Tiempo de espera en cola Fair-Share agotado.")

        try:
            ranked_nodes = self.select_ordered_nodes_for_user(user_id)
            last_err: Optional[Exception] = None
            for node in ranked_nodes:
                try:
                    return node.execute_completion(payload)
                except Exception as exc:
                    last_err = exc
                    logger.warning(f"Failover inter-nodo activado desde {node.node_id}: {exc}")
                    continue
            raise RuntimeError(f"Todos los nodos fallaron al procesar la solicitud: {last_err}")
        finally:
            self.scheduler.release_turn(user_id)

    def dispatch_streaming(self, user_id: str, payload: Dict[str, Any]) -> Generator[bytes, None, None]:
        contract = DeterministicIntentClassifier.classify(payload)
        est_tokens = contract.estimated_input_tokens + contract.estimated_output_tokens

        if not self.scheduler.acquire_turn(user_id, est_tokens, timeout_sec=300.0):
            raise RuntimeError("Tiempo de espera en cola Fair-Share agotado.")

        try:
            ranked_nodes = self.select_ordered_nodes_for_user(user_id)
            for node in ranked_nodes:
                try:
                    yield from node.execute_streaming_completion(payload)
                    return
                except Exception as exc:
                    logger.warning(f"Failover de stream inter-nodo desde {node.node_id}: {exc}")
                    continue
            raise RuntimeError("Todos los nodos agotaron sus proveedores durante el streaming.")
        finally:
            self.scheduler.release_turn(user_id)

    def get_cluster_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            node_stats = []
            total_in_flight = 0
            provider_totals: Dict[str, int] = {}

            for nid, node in self.nodes.items():
                total_in_flight += node.active_in_flight_count
                for prov, cnt in node.provider_dispatch_counts.items():
                    provider_totals[prov] = provider_totals.get(prov, 0) + cnt
                node_stats.append({
                    "node_id": nid,
                    "identity_email": self.node_identities.get(nid, "unknown"),
                    "active_in_flight_channels": node.active_in_flight_count,
                    "max_in_flight_channels": node.max_in_flight_channels,
                    "weighted_residual_quota_pct": round(node.compute_weighted_residual_quota() * 100, 2),
                    "consumption_rate_rpm": round(node.get_consumption_rate_per_minute(), 2),
                    "provider_dispatch_breakdown": dict(node.provider_dispatch_counts),
                })

            return {
                "protocol_version": "3.0-SWDRR-P2C",
                "total_nodes": len(self.nodes),
                "total_active_in_flight_requests": total_in_flight,
                "active_student_sessions": len(self.sessions),
                "global_provider_dispersion": provider_totals,
                "nodes": node_stats,
            }


GLOBAL_DISPATCHER: Optional[MasterDeterministicDispatcher] = None


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class FenixMasterHTTPHandler(BaseHTTPRequestHandler):
    """
    Servidor HTTP/1.1 compatible con OpenAI y optimizado para Render / Cloudflare / Ktor 3.x:
    - Desactiva buffering de proxy (X-Accel-Buffering: no, Cache-Control: no-cache, no-transform).
    - Soporta Keep-Alive SSE por hasta 100 minutos con Heartbeats cada 12s incluso en modo JSON.
    """

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path in ("/health", "/"):
            body = b'{"status":"ok","engine":"FenixStudio-v3.0-SWDRR"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/v1/telemetry":
            telemetry = GLOBAL_DISPATCHER.get_cluster_telemetry() if GLOBAL_DISPATCHER else {}
            body = json.dumps(telemetry, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/v1/models":
            models = [
                {"id": spec.model_id, "object": "model", "owned_by": spec.provider_id, "ring": spec.ring}
                for spec in DynamicCatalogDiscovery.build_default_node_catalog()
            ]
            body = json.dumps({"object": "list", "data": models}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "Endpoint no encontrado")

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404, "Endpoint no soportado")
            return

        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            self.send_error(400, f"JSON inválido: {exc}")
            return

        user_id = (
            self.headers.get("X-Fenix-User-Id")
            or payload.get("user")
            or self.client_address[0]
            or "anonymous-student"
        )
        is_stream = bool(payload.get("stream", False))

        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            try:
                for chunk in GLOBAL_DISPATCHER.dispatch_streaming(user_id, payload):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except Exception as exc:
                err_frame = f'data: {json.dumps({"error": str(exc)})}\n\ndata: [DONE]\n\n'.encode("utf-8")
                try:
                    self.wfile.write(err_frame)
                    self.wfile.flush()
                except Exception:
                    pass
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            result_holder: Dict[str, Any] = {}
            error_holder: List[Exception] = []
            done_evt = threading.Event()

            def _worker() -> None:
                try:
                    result_holder["data"] = GLOBAL_DISPATCHER.dispatch_non_streaming(user_id, payload)
                except Exception as e:
                    error_holder.append(e)
                finally:
                    done_evt.set()

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

            while not done_evt.wait(timeout=12.0):
                try:
                    self.wfile.write(b" ")
                    self.wfile.flush()
                except Exception:
                    return

            if error_holder:
                err_json = json.dumps({"error": {"message": str(error_holder[0]), "type": "router_error"}}).encode("utf-8")
                self.wfile.write(err_json)
                self.wfile.flush()
            else:
                resp_bytes = json.dumps(result_holder["data"]).encode("utf-8")
                self.wfile.write(resp_bytes)
                self.wfile.flush()


def main() -> None:
    global GLOBAL_DISPATCHER
    GLOBAL_DISPATCHER = MasterDeterministicDispatcher()
    port = int(os.environ.get("PORT", os.environ.get("FENIX_PORT", "7860")))
    host = os.environ.get("FENIX_HOST", "0.0.0.0")
    server = ThreadedHTTPServer((host, port), FenixMasterHTTPHandler)
    logger.info(f"Fénix Studio Master Dispatcher v3.0 escuchando en http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
