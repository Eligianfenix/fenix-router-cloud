/**
 * cloudflare_edge_worker.js - Escudo Edge Anycast a Costo $0.00
 * Mantiene vivo el stream SSE del teléfono Android, balancea entre espejos
 * y ejecuta el cron keep-alive cada 14 minutos para evitar suspensión en Render Free.
 */

const UPSTREAM_MIRRORS = [
  "https://fenix-studio-router-v3.onrender.com"
];

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Health check del Worker
    if (url.pathname === "/health") {
      return new Response(JSON.stringify({ 
        status: "edge_ok", 
        pop: request.cf?.colo || "AR",
        timestamp: Date.now() 
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      });
    }

    const bodyBuffer = request.method === "POST" ? await request.arrayBuffer() : null;

    // Dispersión y Failover entre espejos Upstream
    for (const mirrorBase of UPSTREAM_MIRRORS) {
      const targetUrl = `${mirrorBase}${url.pathname}${url.search}`;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 25000);

      try {
        const upstreamReq = new Request(targetUrl, {
          method: request.method,
          headers: request.headers,
          body: bodyBuffer ? bodyBuffer.slice(0) : null,
          signal: controller.signal
        });

        const response = await fetch(upstreamReq);
        clearTimeout(timeoutId);

        if (response.ok || response.status < 500) {
          const respHeaders = new Headers(response.headers);
          respHeaders.set("Cache-Control", "no-cache, no-transform");
          respHeaders.set("X-Accel-Buffering", "no");
          respHeaders.set("X-Fenix-Edge-Mirror", mirrorBase);
          return new Response(response.body, {
            status: response.status,
            headers: respHeaders
          });
        }
      } catch (err) {
        clearTimeout(timeoutId);
        continue;
      }
    }

    return new Response(
      JSON.stringify({ error: { message: "Clúster en mantenimiento temporal, reintente en segundos." } }),
      { status: 503, headers: { "Content-Type": "application/json" } }
    );
  },

  // Cron Keep-Alive cada 14 minutos para evitar suspensión de Render Free
  async scheduled(event, env, ctx) {
    for (const mirror of UPSTREAM_MIRRORS) {
      try {
        await fetch(`${mirror}/health`, { method: "GET" });
      } catch (err) {
        // Silencioso en keep-alive
      }
    }
  }
};
