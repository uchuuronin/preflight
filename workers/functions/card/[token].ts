/** Cloudflare Pages Function, GET /card/<share token>: the PNG share card (see _card/routes.ts). */
import { FONTS, WASM } from "../_card/assets";
import { cardPng } from "../_card/routes";
import type { CardEnv } from "../_card/record";
import { renderPng } from "../_card/render";
import { CARD_W } from "../_card/svg";

const renderer = { png: (svg: string) => renderPng(svg, WASM, FONTS, CARD_W) };

export const onRequestGet = (context: { request: Request; env: CardEnv; params: { token: string } }): Promise<Response> =>
	cardPng(context.request, context.env, String(context.params.token), renderer);
