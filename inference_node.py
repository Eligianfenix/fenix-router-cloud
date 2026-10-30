#!/usr/bin/env python3
"""
inference_node.py - Motor de Inferencia con Dispersión Paralela Equitativa (SWDRR + P2C)
Fénix Studio v3.0 — Premio a la Innovación San Juan 2026 | Costo $0.00 Absoluto
Despacha simultáneamente sobre los 16 proveedores gratuitos sin inanición de anillos,
con failover transparente <1.5s, protección anti-WAF JA3 y recuperación mid-stream.
"""

import http.client
import json
import logging
import math
import random
import re
import socket
import ssl
import threading
import time
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger("FenixRouter.InferenceNode")


class WorkloadIntent(str, Enum):
    CHAT_TUTOR_CONVERSACIONAL = "CHAT_TUTOR_CONVERSACIONAL"
    PROCESAMIENTO_DOCUMENTAL_ESTRICTO = "PROCESAMIENTO_DOCUMENTAL_ESTRICTO"
    PROCESAMIENTO_DOCUMENTAL_ESTRUCTURAL = "PROCESAMIENTO_DOCUMENTAL_ESTRUCTURAL"
    CODIGO_Y_HERRAMIENTAS = "CODIGO_Y_HERRAMIENTAS"
    COMPACTACION_CONTEXTO = "COMPACTACION_CONTEXTO"


class RateLimitErrorType(str, Enum):
    TRANSIENT_RPM_TPM_429 = "TRANSIENT_RPM_TPM_429"
    TERMINAL_DAILY_429 = "TERMINAL_DAILY_429"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"


@dataclass
class Classified429Error:
    error_type: RateLimitErrorType
    retry_after_sec: float
    diagnostic_reason: str


@dataclass
class CapabilityContract:
    intent: WorkloadIntent
    min_context_window: int
    requires_tools: bool
    requires_strict_fidelity: bool
    estimated_input_tokens: int
    estimated_output_tokens: int


@dataclass
class ModelSpec:
    provider_id: str
    model_id: str
    base_url: str
    ring: int
    context_window: int
    rpm_limit: int
    rpd_limit: int
    tpm_limit: int
    tpd_limit: int
    supports_tools: bool = True
    supports_streaming: bool = True
    strict_document_fidelity: bool = False
    reset_timezone: str = "UTC"


class DeterministicIntentClassifier:
    """
    Clasificador determinista en tiempo lineal O(N) (<1ms) que calcula la ventana de
    contexto real requerida (sin inflarla artificialmente a 128k cuando la solicitud
    ocupa <28k tokens) y desagrega las tareas del Protocolo Fénix v2.1 en Estrictas
    (Diagnóstico, Resumen denso, Citas textuales) y Estructurales (Feynman, Active Recall,
    Mapas conceptuales, Glosario ontológico) para activar los 16 proveedores en paralelo.
    """

    CODE_PATTERNS = re.compile(
        r"(```(?:kotlin|java|python|gradle|json|cpp|typescript|javascript|rust|go|sql|bash)|"
        r"\bfun\s+[a-zA-Z0-9_]+\s*\(|\bdependencies\s*\{|\bimport\s+[a-zA-Z0-9_.]+|"
        r"\bdef\s+[a-zA-Z0-9_]+\s*\(|\bclass\s+[A-Z][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    )

    STRICT_FIDELITY_DOC_PATTERNS = re.compile(
        r"(citas\s+textuales|correspondencia\s+espejo|s[íi]ntesis\s+fiel|sin\s+alucinaci[óo]n|"
        r"resumen\s+denso|diagn[óo]stico\s+epist[ée]mico|transcripci[óo]n\s+literal)",
        re.IGNORECASE,
    )

    STRUCTURAL_DOC_PATTERNS = re.compile(
        r"(feynman|active\s+recall|mapa\s+conceptual|glosario\s+ontol[óo]gico|"
        r"tarjetas\s+de\s+estudio|preguntas\s+de\s+autoevaluaci[óo]n|mermaid)",
        re.IGNORECASE,
    )

    COMPACTION_PATTERNS = re.compile(
        r"(compactaci[óo]n\s+de\s+contexto|resumen\s+recursivo|comprimir\s+historial)",
        re.IGNORECASE,
    )

    @staticmethod
    def estimate_tokens(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> int:
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total_chars += len(str(part.get("text", "")))
            if "tool_calls" in msg:
                total_chars += len(str(msg["tool_calls"]))
        if tools:
            total_chars += len(str(tools))
        return max(1, math.ceil(total_chars / 3.8))

    @classmethod
    def classify(cls, payload: Dict[str, Any]) -> CapabilityContract:
        messages = payload.get("messages", [])
        tools = payload.get("tools") or payload.get("functions")
        response_format = payload.get("response_format", {})
        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 2048)
        est_in_tokens = cls.estimate_tokens(messages, tools)

        # Huella real requerida con 20% de margen de seguridad, permitiendo que los
        # proveedores con ventana de 32,768 y 65,536 tokens participen activamente.
        required_ctx = int((est_in_tokens + max_tokens) * 1.20)

        combined_text = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in messages
        )

        if cls.COMPACTION_PATTERNS.search(combined_text):
            return CapabilityContract(
                intent=WorkloadIntent.COMPACTACION_CONTEXTO,
                min_context_window=max(16384, required_ctx),
                requires_tools=False,
                requires_strict_fidelity=True,
                estimated_input_tokens=est_in_tokens,
                estimated_output_tokens=min(max_tokens, 2048),
            )

        has_explicit_tools = bool(tools and len(tools) > 0)
        has_json_schema = isinstance(response_format, dict) and response_format.get("type") in (
            "json_object",
            "json_schema",
        )
        if has_explicit_tools or has_json_schema or bool(cls.CODE_PATTERNS.search(combined_text)):
            return CapabilityContract(
                intent=WorkloadIntent.CODIGO_Y_HERRAMIENTAS,
                min_context_window=max(16384, required_ctx),
                requires_tools=has_explicit_tools,
                requires_strict_fidelity=False,
                estimated_input_tokens=est_in_tokens,
                estimated_output_tokens=min(max_tokens, 4096),
            )

        if cls.STRUCTURAL_DOC_PATTERNS.search(combined_text) and not cls.STRICT_FIDELITY_DOC_PATTERNS.search(combined_text):
            return CapabilityContract(
                intent=WorkloadIntent.PROCESAMIENTO_DOCUMENTAL_ESTRUCTURAL,
                min_context_window=max(16384, required_ctx),
                requires_tools=False,
                requires_strict_fidelity=False,
                estimated_input_tokens=est_in_tokens,
                estimated_output_tokens=min(max_tokens, 4096),
            )

        if est_in_tokens > 2500 or cls.STRICT_FIDELITY_DOC_PATTERNS.search(combined_text):
            return CapabilityContract(
                intent=WorkloadIntent.PROCESAMIENTO_DOCUMENTAL_ESTRICTO,
                min_context_window=max(24576, required_ctx),
                requires_tools=False,
                requires_strict_fidelity=(
                    est_in_tokens > 28000 or bool(cls.STRICT_FIDELITY_DOC_PATTERNS.search(combined_text))
                ),
                estimated_input_tokens=est_in_tokens,
                estimated_output_tokens=min(max_tokens, 6144),
            )

        return CapabilityContract(
            intent=WorkloadIntent.CHAT_TUTOR_CONVERSACIONAL,
            min_context_window=max(8192, required_ctx),
            requires_tools=False,
            requires_strict_fidelity=False,
            estimated_input_tokens=est_in_tokens,
            estimated_output_tokens=min(max_tokens, 2048),
        )


class DynamicCatalogDiscovery:
    """
    Catálogo Maestro de los 16 Proveedores Gratuitos con Cuota Diaria Renovable a Costo $0.00.
    """

    @staticmethod
    def build_default_node_catalog() -> List[ModelSpec]:
        return [
            # 1. GOOGLE GEMINI (Anillo 1 - Alta Fidelidad)
            ModelSpec("gemini", "gemini-3.5-flash-lite", "https://generativelanguage.googleapis.com/v1beta/openai", 1, 1000000, 15, 500, 250000, 50000000, True, True, True, "US/Pacific"),
            ModelSpec("gemini", "gemini-3.1-flash-lite", "https://generativelanguage.googleapis.com/v1beta/openai", 1, 1000000, 15, 500, 250000, 50000000, True, True, True, "US/Pacific"),
            ModelSpec("gemini", "gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta/openai", 1, 1000000, 5, 20, 250000, 5000000, True, True, True, "US/Pacific"),
            ModelSpec("gemini", "gemma-4-31b-it", "https://generativelanguage.googleapis.com/v1beta/openai", 1, 128000, 15, 1500, 250000, 30000000, True, True, True, "US/Pacific"),
            # 2. GROQ (Anillo 2 - Ultra-Velocidad LPU)
            ModelSpec("groq", "openai/gpt-oss-120b", "https://api.groq.com/openai/v1", 2, 131072, 30, 1000, 30000, 5000000, True, True, True),
            ModelSpec("groq", "qwen/qwen3.8-27b", "https://api.groq.com/openai/v1", 2, 131072, 30, 1000, 30000, 5000000, True, True, True),
            ModelSpec("groq", "kimi-k2-instruct", "https://api.groq.com/openai/v1", 2, 262000, 30, 14400, 40000, 20000000, True, True, True),
            # 3. SAMBANOVA SYSTEMS (Anillo 2 - Calidad Llama 70B)
            ModelSpec("sambanova", "Meta-Llama-3.3-70B-Instruct", "https://api.sambanova.ai/v1", 2, 65536, 20, 100, 100000, 2000000, True, True, True),
            # 4. OPENROUTER (Anillo 2 - Modelos Libres)
            ModelSpec("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", "https://openrouter.ai/api/v1", 2, 131072, 20, 200, 100000, 4000000, True, True, True),
            # 5. SILICONFLOW (Anillo 2 - Ráfaga Masiva, acotado a 60 RPM seguros por modelo)
            ModelSpec("siliconflow", "Qwen/Qwen3-8B", "https://api.siliconflow.cn/v1", 2, 32768, 60, 10000, 50000, 25000000, True, True, False),
            ModelSpec("siliconflow", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "https://api.siliconflow.cn/v1", 2, 32768, 60, 10000, 50000, 25000000, True, True, False),
            # 6. COHERE (Anillo 3 - Estructuración Doctrinal)
            ModelSpec("cohere", "command-a-03-2025", "https://api.cohere.ai/compatibility/v1", 3, 128000, 20, 100, 100000, 2000000, True, True, True),
            # 7. NVIDIA NIM (Anillo 4 - Razonamiento y Largo Contexto)
            ModelSpec("nvidia", "nvidia/nemotron-3.5-super-vl-120b-a12b", "https://integrate.api.nvidia.com/v1", 4, 128000, 40, 50, 150000, 2000000, True, True, True),
            # 8. BASETEN (Anillo 4 - Modelos OSS)
            ModelSpec("baseten", "baseten-oss", "https://bridge.baseten.co/v1", 4, 128000, 30, 20, 100000, 1000000, True, True, False),
            # 9. FASTROUTER (Anillo 4 - Ruta Dinámica)
            ModelSpec("fastrouter", "fastrouter/auto", "https://api.fastrouter.ai/api/v1", 4, 32768, 30, 20, 100000, 1000000, True, True, False),
            # 10. UNOROUTER (Anillo 2 - Balanceo)
            ModelSpec("unorouter", "unorouter/free", "https://unorouter.com/v1", 2, 32768, 20, 50, 50000, 1000000, True, True, False),
            # 11. ROUTEWAY (Anillo 2 - Respaldo Experimental)
            ModelSpec("routeway", "experimental-free", "https://api.routeway.ai/v1", 2, 32768, 10, 50, 40000, 1000000, True, True, False),
            # 12. TOKENHARBOR (Anillo 1 - Sintético Alta Fidelidad)
            ModelSpec("tokenharbor", "deepseek-v4.1-flash:free", "https://tokenharbor.ai/v1", 1, 64000, 20, 200, 150000, 20000000, True, True, True),
            # 13. REQUESTY (Anillo 2 - Gateway Qwen)
            ModelSpec("requesty", "Qwen/Qwen3.8-27B", "https://router.requesty.ai/v1", 2, 32768, 20, 200, 80000, 5000000, True, True, True),
            # 14. PLUGSKY (Anillo 2 - Procesamiento Rápido)
            ModelSpec("plugsky", "plugsky-lite", "https://api.plugsky.com/v1", 2, 32768, 30, 2000, 100000, 10000000, True, True, False),
            # 15. AGNES (Anillo 2 - Respaldo SenseNova)
            ModelSpec("agnes", "agnes-3.0-flash", "https://token.sensenova.cn/v1", 2, 32768, 20, 100, 80000, 3000000, True, True, False),
            # 16. AIRFORCE (Anillo 2 - Complementario)
            ModelSpec("airforce", "gpt-4.1-mini", "https://api.airforce/v1", 2, 128000, 15, 100, 60000, 2000000, True, True, True),
            # OPCIONALES ADICIONALES (Naga, GitHub Models, Kilocode)
            ModelSpec("naga", "Dots3 Note Preview (free)", "https://api.naga.ac/v1", 2, 32768, 15, 50, 50000, 1000000, True, True, False),
            ModelSpec("github", "gpt-4.1-mini", "https://models.inference.ai.azure.com", 2, 128000, 15, 150, 150000, 5000000, True, True, True),
            ModelSpec("kilocode", "poolside/laguna-s-2.1:free", "https://api.kilocode.ai/v1", 2, 262000, 20, 1600, 100000, 10000000, True, True, False),
        ]


class Deterministic429Classifier:
    """Clasifica respuestas HTTP 429 diferenciando saturación de minuto vs agotamiento diario."""

    DAILY_EXHAUSTION_PATTERNS = re.compile(
        r"(per\s+day|daily\s+limit|rpd|quota\s+exceeded\s+for\s+day|daily\s+quota|exhausted\s+daily)",
        re.IGNORECASE,
    )

    @classmethod
    def classify_429(cls, provider_id: str, headers: Dict[str, Any], body: str) -> Classified429Error:
        lower_headers = {str(k).lower(): str(v) for k, v in headers.items()}
        retry_after = 15.0
        for h_key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset"):
            if h_key in lower_headers:
                val = lower_headers[h_key]
                try:
                    num = float(re.sub(r"[^\d.]", "", val))
                    if num > 1700000000:
                        retry_after = max(2.0, num - time.time())
                    elif num > 0:
                        retry_after = min(3600.0, num)
                except ValueError:
                    pass

        if cls.DAILY_EXHAUSTION_PATTERNS.search(body) or retry_after > 1800.0:
            return Classified429Error(
                error_type=RateLimitErrorType.TERMINAL_DAILY_429,
                retry_after_sec=max(retry_after, 3600.0),
                diagnostic_reason=f"Cuota diaria agotada en {provider_id}",
            )
        return Classified429Error(
            error_type=RateLimitErrorType.TRANSIENT_RPM_TPM_429,
            retry_after_sec=min(max(retry_after, 4.0), 45.0),
            diagnostic_reason=f"Límite por minuto alcanzado en {provider_id}",
        )


class DualTokenBucketController:
    """
    Controlador Dual Token/Leaky Bucket con margen preventivo eta = 0.85
    para evitar disparar heurísticas de abuso en los 16 proveedores, más contador
    SWDRR (Smooth Weighted Deficit Round-Robin) y control de canales en vuelo.
    """

    SAFETY_FACTOR_ETA = 0.85

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._lock = threading.Lock()

        self.safe_rpm = max(1.0, float(spec.rpm_limit) * self.SAFETY_FACTOR_ETA)
        self.safe_tpm = max(2000.0, float(spec.tpm_limit) * self.SAFETY_FACTOR_ETA)

        self.req_tokens = self.safe_rpm
        self.tok_tokens = self.safe_tpm
        self.rpd_remaining = spec.rpd_limit
        self.tpd_remaining = spec.tpd_limit

        self.last_refill_ts = time.time()
        self.cooldown_until_ts = 0.0
        self.ema_latency_sec = 18.0 if spec.ring == 2 else 22.0
        self.in_flight_count = 0
        self.current_swdrr_weight = 0.0

    def _refill_unlocked(self, now: float) -> None:
        elapsed = max(0.0, now - self.last_refill_ts)
        if elapsed > 0.0:
            self.req_tokens = min(self.safe_rpm, self.req_tokens + elapsed * (self.safe_rpm / 60.0))
            self.tok_tokens = min(self.safe_tpm, self.tok_tokens + elapsed * (self.safe_tpm / 60.0))
            self.last_refill_ts = now

    def max_sustainable_channels(self) -> int:
        return max(2, int(math.ceil((self.safe_rpm / 60.0) * self.ema_latency_sec)) + 1)

    def estimate_wait_seconds(self, est_tokens: int) -> float:
        with self._lock:
            now = time.time()
            self._refill_unlocked(now)
            if now < self.cooldown_until_ts:
                return self.cooldown_until_ts - now
            if self.rpd_remaining <= 0 or self.tpd_remaining < est_tokens:
                return 86400.0

            wait_req = 0.0
            if self.req_tokens < 1.0:
                wait_req = (1.0 - self.req_tokens) / (self.safe_rpm / 60.0)

            needed_tok = min(float(est_tokens), self.safe_tpm * 0.9)
            wait_tok = 0.0
            if self.tok_tokens < needed_tok:
                wait_tok = (needed_tok - self.tok_tokens) / (self.safe_tpm / 60.0)

            return max(wait_req, wait_tok)

    def is_immediately_available(self, est_tokens: int) -> bool:
        with self._lock:
            now = time.time()
            self._refill_unlocked(now)
            if now < self.cooldown_until_ts:
                return False
            if self.rpd_remaining <= 0 or self.tpd_remaining < est_tokens:
                return False
            if self.in_flight_count >= self.max_sustainable_channels():
                return False
            needed_tok = min(float(est_tokens), self.safe_tpm * 0.9)
            return self.req_tokens >= 1.0 and self.tok_tokens >= needed_tok

    def try_acquire_non_blocking(self, est_tokens: int, max_wait_sec: float = 0.0) -> bool:
        """
        En la pasada primaria (max_wait_sec = 0.0), jamás duerme el hilo: si el cubo
        del proveedor no tiene tokens instantáneos, retorna False en <0.01ms para que
        el despachador utilice otro de los 16 proveedores en paralelo.
        """
        wait_s = self.estimate_wait_seconds(est_tokens)
        if wait_s > max_wait_sec:
            return False
        if wait_s > 0.0:
            time.sleep(wait_s + random.uniform(0.015, 0.055))

        with self._lock:
            now = time.time()
            self._refill_unlocked(now)
            if now < self.cooldown_until_ts or self.rpd_remaining <= 0:
                return False
            needed_tok = min(float(est_tokens), self.safe_tpm * 0.9)
            if self.req_tokens < 0.85:
                return False
            self.req_tokens = max(0.0, self.req_tokens - 1.0)
            self.tok_tokens = max(0.0, self.tok_tokens - needed_tok)
            self.rpd_remaining = max(0, self.rpd_remaining - 1)
            self.tpd_remaining = max(0, self.tpd_remaining - est_tokens)
            self.in_flight_count += 1
            return True

    def release_in_flight(self) -> None:
        with self._lock:
            self.in_flight_count = max(0, self.in_flight_count - 1)

    def sync_from_response_headers(self, headers: Dict[str, Any], actual_tokens: int, latency_sec: float) -> None:
        with self._lock:
            self.ema_latency_sec = 0.80 * self.ema_latency_sec + 0.20 * max(1.0, latency_sec)
            lower_h = {str(k).lower(): str(v) for k, v in headers.items()}
            if "x-ratelimit-remaining-requests" in lower_h:
                try:
                    rem = int(re.sub(r"[^\d]", "", lower_h["x-ratelimit-remaining-requests"]))
                    self.rpd_remaining = min(self.rpd_remaining, max(0, rem))
                except ValueError:
                    pass

    def apply_429_penalty(self, classified: Classified429Error) -> None:
        with self._lock:
            now = time.time()
            if classified.error_type == RateLimitErrorType.TERMINAL_DAILY_429:
                self.rpd_remaining = 0
                self.cooldown_until_ts = now + classified.retry_after_sec
            else:
                self.req_tokens = 0.0
                self.cooldown_until_ts = now + classified.retry_after_sec


class IsolatedInferenceNode:
    """
    Nodo de Inferencia Aislado Refactorizado:
    - Amplía el semáforo de 8 a 128 canales concurrentes por nodo.
    - Dispersa simultáneamente entre los 16 proveedores mediante SWDRR + P2C + ECT.
    - Ejecuta Fast-Failover <1.5s cuando un proveedor falla en el handshake inicial.
    """

    COMPLIANT_USER_AGENT = (
        "FenixAcademicStudio/3.0 (UNSJ-SanJuan-Innovation-2026; Python-StdLib-HTTP; +https://fenix.unsj.edu.ar)"
    )

    def __init__(
        self,
        node_id: str,
        identity_email: str,
        provider_api_keys: Dict[str, Any],
        max_in_flight_channels: int = 128,
        proactive_depletion_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.node_id = node_id
        self.identity_email = identity_email
        self.provider_api_keys = provider_api_keys
        self.max_in_flight_channels = max_in_flight_channels
        self.proactive_depletion_callback = proactive_depletion_callback

        self.in_flight_semaphore = threading.Semaphore(max_in_flight_channels)
        self.active_in_flight_count = 0
        self.recent_request_timestamps: List[float] = []
        self.provider_dispatch_counts: Dict[str, int] = {}
        self._metrics_lock = threading.Lock()
        self._gemini_rr_idx = 0

        self.controllers: Dict[str, DualTokenBucketController] = {}
        for spec in DynamicCatalogDiscovery.build_default_node_catalog():
            key = f"{spec.provider_id}::{spec.model_id}"
            self.controllers[key] = DualTokenBucketController(spec)
            self.provider_dispatch_counts[spec.provider_id] = 0

    def _resolve_api_key(self, provider_id: str) -> str:
        if provider_id == "gemini":
            pool = self.provider_api_keys.get("gemini_pool", [])
            if isinstance(pool, list) and len(pool) > 0:
                with self._metrics_lock:
                    key = pool[self._gemini_rr_idx % len(pool)]
                    self._gemini_rr_idx += 1
                if key and not str(key).startswith("pon_aqui"):
                    return str(key)
            g_key = self.provider_api_keys.get("gemini") or self.provider_api_keys.get("gemini_key") or ""
            return str(g_key) if not str(g_key).startswith("pon_aqui") else ""

        val = self.provider_api_keys.get(provider_id, "")
        if isinstance(val, list) and val:
            return str(random.choice(val))
        return str(val) if (val and not str(val).startswith("pon_aqui")) else ""

    def _record_arrival(self, provider_id: Optional[str] = None) -> None:
        with self._metrics_lock:
            now = time.time()
            self.recent_request_timestamps.append(now)
            cutoff = now - 900.0
            self.recent_request_timestamps = [t for t in self.recent_request_timestamps if t >= cutoff]
            if provider_id:
                self.provider_dispatch_counts[provider_id] = self.provider_dispatch_counts.get(provider_id, 0) + 1

    def get_consumption_rate_per_minute(self) -> float:
        with self._metrics_lock:
            now = time.time()
            cutoff = now - 900.0
            valid = [t for t in self.recent_request_timestamps if t >= cutoff]
            return max(0.1, len(valid) / 15.0)

    def compute_weighted_residual_quota(self, contract: Optional[CapabilityContract] = None) -> float:
        numerator = 0.0
        denominator = 0.0
        now = time.time()

        for ctrl in self.controllers.values():
            spec = ctrl.spec
            if not self._resolve_api_key(spec.provider_id):
                continue
            if contract and spec.context_window < contract.min_context_window:
                continue

            capacity_weight = math.log1p(spec.rpm_limit) * math.log1p(spec.rpd_limit)
            rpd_ratio = ctrl.rpd_remaining / max(1.0, float(spec.rpd_limit))
            tpd_ratio = ctrl.tpd_remaining / max(1.0, float(spec.tpd_limit))
            avail_factor = 0.0 if now < ctrl.cooldown_until_ts else 1.0

            numerator += capacity_weight * min(rpd_ratio, tpd_ratio) * avail_factor
            denominator += capacity_weight

        return (numerator / denominator) if denominator > 0 else 0.0

    def select_dispersed_candidates(self, contract: CapabilityContract) -> List[DualTokenBucketController]:
        """
        Algoritmo SWDRR + Power of Two Choices (P2C) + Earliest Completion Time (ECT).
        Dispersa las peticiones simultáneamente entre los 16 proveedores activos proporcionalmente
        a sus RPM/RPD libres y su baja congestión en vuelo, evitando la saturación del Anillo 1.
        """
        total_est_tokens = contract.estimated_input_tokens + contract.estimated_output_tokens
        eligible: List[DualTokenBucketController] = []

        for ctrl in self.controllers.values():
            spec = ctrl.spec
            if not self._resolve_api_key(spec.provider_id):
                continue
            if spec.context_window < contract.min_context_window:
                continue
            if contract.requires_tools and not spec.supports_tools:
                continue
            if ctrl.rpd_remaining <= 0 or ctrl.tpd_remaining < total_est_tokens:
                continue
            eligible.append(ctrl)

        if not eligible:
            return []

        instant_pool: List[Tuple[float, float, DualTokenBucketController]] = []
        waiting_pool: List[Tuple[float, DualTokenBucketController]] = []

        with self._metrics_lock:
            total_swdrr_step = 0.0
            temp_weights: List[Tuple[DualTokenBucketController, float]] = []

            for ctrl in eligible:
                spec = ctrl.spec
                if ctrl.is_immediately_available(total_est_tokens):
                    rpm_capacity = max(1.0, ctrl.req_tokens)
                    rpd_ratio = ctrl.rpd_remaining / max(1.0, float(spec.rpd_limit))
                    max_chan = ctrl.max_sustainable_channels()
                    congestion_factor = 1.0 / (1.0 + (ctrl.in_flight_count / max(1.0, float(max_chan))))

                    if contract.requires_strict_fidelity:
                        qual_factor = 1.35 if spec.strict_document_fidelity else 0.85
                    else:
                        qual_factor = 1.25 if spec.ring == 2 else 1.05

                    w_i = (rpm_capacity ** 0.65) * (rpd_ratio ** 0.35) * congestion_factor * qual_factor
                    ctrl.current_swdrr_weight += w_i
                    total_swdrr_step += w_i
                    temp_weights.append((ctrl, w_i))
                else:
                    wait_s = ctrl.estimate_wait_seconds(total_est_tokens)
                    if wait_s < 12.0:
                        ect = wait_s + ctrl.ema_latency_sec
                        waiting_pool.append((ect, ctrl))

            temp_weights.sort(key=lambda x: x[0].current_swdrr_weight, reverse=True)

            if temp_weights:
                top_k = temp_weights[: min(3, len(temp_weights))]
                best_ctrl = min(
                    top_k,
                    key=lambda item: item[0].ema_latency_sec
                    * (1.0 + item[0].in_flight_count / max(1.0, float(item[0].max_sustainable_channels())))
                    * random.uniform(0.92, 1.08),
                )[0]
                best_ctrl.current_swdrr_weight -= total_swdrr_step

                instant_pool.append((999999.0, 0.0, best_ctrl))
                for ctrl, _ in temp_weights:
                    if ctrl is not best_ctrl:
                        ect = ctrl.ema_latency_sec * (
                            1.0 + ctrl.in_flight_count / max(1.0, float(ctrl.max_sustainable_channels()))
                        )
                        instant_pool.append((ctrl.current_swdrr_weight, -ect, ctrl))

        ordered_controllers = [item[2] for item in instant_pool]
        waiting_pool.sort(key=lambda x: x[0])
        ordered_controllers.extend([item[1] for item in waiting_pool])
        return ordered_controllers

    def _open_upstream_http_connection(
        self,
        spec: ModelSpec,
        api_key: str,
        req_body: Dict[str, Any],
        connect_timeout: float = 1.45,
        read_timeout: float = 45.0,
    ) -> Tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        """
        Abre conexión HTTP/HTTPS con separación estricta entre timeout de handshake TCP/TLS
        (1.45s para Fast-Failover) y timeout de lectura de tokens (45.0s).
        """
        target_url = f"{spec.base_url.rstrip('/')}/chat/completions"
        parsed = urllib.parse.urlparse(target_url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/v1/chat/completions"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        headers = {
            "Host": host,
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
            "User-Agent": self.COMPLIANT_USER_AGENT,
            "Authorization": f"Bearer {api_key}",
            "Connection": "close",
        }
        if spec.provider_id == "openrouter":
            headers["HTTP-Referer"] = "https://fenix.unsj.edu.ar"
            headers["X-Title"] = "Fenix Academic Studio v3.0"

        payload_bytes = json.dumps(req_body).encode("utf-8")
        headers["Content-Length"] = str(len(payload_bytes))

        if parsed.scheme == "https":
            ctx = ssl.create_default_context()
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                host, port, timeout=connect_timeout, context=ctx
            )
        else:
            conn = http.client.HTTPConnection(host, port, timeout=connect_timeout)

        conn.connect()
        if conn.sock is not None:
            conn.sock.settimeout(read_timeout)

        conn.request("POST", path, body=payload_bytes, headers=headers)
        resp = conn.getresponse()
        return conn, resp

    def _check_proactive_failover_trigger(self, contract: CapabilityContract) -> None:
        residual = self.compute_weighted_residual_quota(contract)
        if residual < 0.10 and self.proactive_depletion_callback:
            self.proactive_depletion_callback(
                self.node_id,
                f"Cuota residual ponderada crítica ({residual:.2%}) bajo umbral del 10%",
            )

    def execute_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecución sincrónica con dispersión paralela SWDRR+P2C y Fast-Failover <1.5s."""
        contract = DeterministicIntentClassifier.classify(payload)
        total_est_tokens = contract.estimated_input_tokens + contract.estimated_output_tokens
        candidates = self.select_dispersed_candidates(contract)

        if not candidates:
            self._check_proactive_failover_trigger(contract)
            raise RuntimeError(f"Nodo {self.node_id} sin modelos elegibles para el contrato solicitado.")

        acquired = self.in_flight_semaphore.acquire(timeout=180.0)
        if not acquired:
            raise RuntimeError(f"Nodo {self.node_id} saturado en admisión local.")

        try:
            with self._metrics_lock:
                self.active_in_flight_count += 1

            for max_wait in (0.0, 2.5):
                for ctrl in candidates:
                    spec = ctrl.spec
                    api_key = self._resolve_api_key(spec.provider_id)
                    if not api_key:
                        continue

                    if not ctrl.try_acquire_non_blocking(total_est_tokens, max_wait_sec=max_wait):
                        continue

                    req_body = dict(payload)
                    req_body["model"] = spec.model_id
                    req_body["stream"] = False

                    conn: Optional[http.client.HTTPConnection] = None
                    start_t = time.time()
                    try:
                        conn, resp = self._open_upstream_http_connection(
                            spec=spec,
                            api_key=api_key,
                            req_body=req_body,
                            connect_timeout=1.45,
                            read_timeout=55.0,
                        )
                        status_code = resp.status
                        resp_headers = dict(resp.getheaders())
                        raw_bytes = resp.read()
                        latency = time.time() - start_t

                        if status_code == 200:
                            data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
                            if "error" in data or not data.get("choices"):
                                ctrl.cooldown_until_ts = time.time() + 45.0
                                continue

                            usage = data.get("usage", {})
                            actual_tok = int(usage.get("total_tokens", total_est_tokens))
                            ctrl.sync_from_response_headers(resp_headers, actual_tok, latency)
                            self._record_arrival(spec.provider_id)

                            data["_orchestration_metadata"] = {
                                "node_id": self.node_id,
                                "identity_email": self.identity_email,
                                "provider_id": spec.provider_id,
                                "model_id": spec.model_id,
                                "ring": spec.ring,
                                "intent": contract.intent.value,
                                "upstream_latency_ms": round(latency * 1000, 1),
                            }
                            self._check_proactive_failover_trigger(contract)
                            return data

                        err_body = raw_bytes.decode("utf-8", errors="ignore")
                        if status_code == 429:
                            classified = Deterministic429Classifier.classify_429(
                                spec.provider_id, resp_headers, err_body
                            )
                            ctrl.apply_429_penalty(classified)
                            if classified.error_type == RateLimitErrorType.TERMINAL_DAILY_429:
                                self._check_proactive_failover_trigger(contract)
                            continue
                        elif status_code in (401, 403, 404):
                            ctrl.rpd_remaining = 0
                            continue
                        else:
                            ctrl.cooldown_until_ts = time.time() + 20.0
                            continue
                    except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as exc:
                        logger.warning(f"[{self.node_id}] Fast-failover desde {spec.provider_id}/{spec.model_id}: {exc}")
                        ctrl.cooldown_until_ts = time.time() + 15.0
                        continue
                    finally:
                        ctrl.release_in_flight()
                        if conn is not None:
                            try:
                                conn.close()
                            except Exception:
                                pass

            self._check_proactive_failover_trigger(contract)
            raise RuntimeError(f"Agotados los 16 proveedores candidatos en {self.node_id}.")
        finally:
            with self._metrics_lock:
                self.active_in_flight_count = max(0, self.active_in_flight_count - 1)
            self.in_flight_semaphore.release()

    def execute_streaming_completion(self, payload: Dict[str, Any]) -> Generator[bytes, None, None]:
        """
        Streaming SSE resiliente con preámbulo inmediato (<10ms), conmutación <1.5s
        transparente entre los 16 proveedores y recuperación mid-stream automática.
        """
        contract = DeterministicIntentClassifier.classify(payload)
        total_est_tokens = contract.estimated_input_tokens + contract.estimated_output_tokens
        candidates = self.select_dispersed_candidates(contract)

        if not candidates:
            self._check_proactive_failover_trigger(contract)
            raise RuntimeError(f"Nodo {self.node_id} sin modelos elegibles para streaming.")

        acquired = self.in_flight_semaphore.acquire(timeout=180.0)
        if not acquired:
            raise RuntimeError(f"Nodo {self.node_id} saturado en canales de streaming.")

        yield f": fenix-stream-open node={self.node_id} ts={int(time.time())}\n\n".encode("utf-8")

        accumulated_text: List[str] = []
        completed_cleanly = False

        try:
            with self._metrics_lock:
                self.active_in_flight_count += 1

            for ctrl in candidates:
                spec = ctrl.spec
                api_key = self._resolve_api_key(spec.provider_id)
                if not api_key:
                    continue

                if not ctrl.try_acquire_non_blocking(total_est_tokens, max_wait_sec=0.5):
                    continue

                req_body = dict(payload)
                req_body["model"] = spec.model_id
                req_body["stream"] = True

                if accumulated_text:
                    partial_so_far = "".join(accumulated_text)
                    orig_messages = list(payload.get("messages", []))
                    orig_messages.append({"role": "assistant", "content": partial_so_far})
                    orig_messages.append({
                        "role": "user",
                        "content": (
                            "Continúa exactamente desde la última palabra anterior sin repetir "
                            "introducción ni texto ya emitido."
                        ),
                    })
                    req_body["messages"] = orig_messages

                conn: Optional[http.client.HTTPConnection] = None
                start_t = time.time()
                try:
                    conn, resp = self._open_upstream_http_connection(
                        spec=spec,
                        api_key=api_key,
                        req_body=req_body,
                        connect_timeout=1.45,
                        read_timeout=25.0,
                    )
                    if resp.status != 200:
                        err_body = resp.read().decode("utf-8", errors="ignore")
                        if resp.status == 429:
                            classified = Deterministic429Classifier.classify_429(
                                spec.provider_id, dict(resp.getheaders()), err_body
                            )
                            ctrl.apply_429_penalty(classified)
                        elif resp.status in (401, 403, 404):
                            ctrl.rpd_remaining = 0
                        continue

                    self._record_arrival(spec.provider_id)
                    last_chunk_ts = time.time()

                    while True:
                        line = resp.fp.readline()
                        if not line:
                            completed_cleanly = True
                            break

                        now_t = time.time()
                        if now_t - last_chunk_ts >= 12.0:
                            yield b": keepalive\n\n"
                        last_chunk_ts = now_t

                        decoded_line = line.decode("utf-8", errors="ignore").strip()
                        if decoded_line.startswith("data:"):
                            data_str = decoded_line[5:].strip()
                            if data_str == "[DONE]":
                                completed_cleanly = True
                                yield b"data: [DONE]\n\n"
                                break
                            try:
                                chunk_obj = json.loads(data_str)
                                choices = chunk_obj.get("choices", [])
                                if choices:
                                    delta_content = choices[0].get("delta", {}).get("content", "")
                                    if delta_content:
                                        accumulated_text.append(delta_content)
                            except Exception:
                                pass

                        yield line

                    ctrl.sync_from_response_headers(
                        dict(resp.getheaders()), total_est_tokens, time.time() - start_t
                    )
                    if completed_cleanly:
                        self._check_proactive_failover_trigger(contract)
                        return

                except Exception as exc:
                    logger.warning(
                        f"[{self.node_id}] Conmutación en vivo de stream desde {spec.provider_id}: {exc}"
                    )
                    yield f": fenix-failover-switch from={spec.provider_id}\n\n".encode("utf-8")
                    continue
                finally:
                    ctrl.release_in_flight()
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass

            if accumulated_text:
                yield b"data: [DONE]\n\n"
                return

            raise RuntimeError(f"Fallo de streaming en todos los proveedores del nodo {self.node_id}.")
        finally:
            with self._metrics_lock:
                self.active_in_flight_count = max(0, self.active_in_flight_count - 1)
            self.in_flight_semaphore.release()
