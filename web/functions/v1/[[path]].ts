/** Cloudflare Pages Function: every /v1/* request goes to the API (see _lib/proxy.ts). */
import { proxy, type ProxyEnv } from "../_lib/proxy";

export const onRequest = (context: { request: Request; env: ProxyEnv }): Promise<Response> => proxy(context.request, context.env);
