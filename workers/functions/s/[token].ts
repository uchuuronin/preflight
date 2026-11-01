/** Cloudflare Pages Function, GET /s/<share token>: the share page (see _card/routes.ts). */
import { sharePage } from "../_card/routes";
import type { CardEnv } from "../_card/record";

export const onRequestGet = (context: { request: Request; env: CardEnv; params: { token: string } }): Promise<Response> =>
	sharePage(context.request, context.env, String(context.params.token));
