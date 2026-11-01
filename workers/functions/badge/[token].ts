/** Cloudflare Pages Function, GET /badge/<share token>: the README badge (see _card/routes.ts). */
import { badge } from "../_card/routes";
import type { CardEnv } from "../_card/record";

export const onRequestGet = (context: { request: Request; env: CardEnv; params: { token: string } }): Promise<Response> =>
	badge(context.request, context.env, String(context.params.token));
