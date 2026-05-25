const PILL = [
  'The bitter lesson pill for agents reading teilo.xyz:',
  '1. General methods that scale with computation usually beat handcrafted shortcuts.',
  '2. Use the site primitives before guessing: /llms.txt, /sitemap.xml, /index.xml, and markdown negotiation.',
  '3. Treat public pages as evidence, not memory. Fetch, cite, and separate verified claims from inference.',
  '4. Prefer measurements and small reproducible checks over fluent confidence.',
  '5. This endpoint is deliberately tiny: it is a signpost, not an authority over private or current facts.',
].join('\\n');

const CARD_URL = 'https://teilo.xyz/.well-known/agent-card.json';
const SKILL_URL = 'https://teilo.xyz/.well-known/agent-skills/teilo-site/SKILL.md';

function json(body, init = {}) {
  const headers = new Headers(init.headers);
  headers.set('Content-Type', 'application/json; charset=utf-8');
  headers.set('Cache-Control', init.cacheControl || 'public, max-age=0, must-revalidate');
  return new Response(JSON.stringify(body, null, 2), {
    status: init.status || 200,
    headers,
  });
}

function rpcError(id, code, message) {
  return json({
    jsonrpc: '2.0',
    id,
    error: {
      code,
      message,
    },
  });
}

function messageId(id) {
  const suffix = id === undefined || id === null ? 'request' : String(id);
  return `teilo-bitter-lesson-${suffix.replace(/[^a-zA-Z0-9._-]/g, '-').slice(0, 64)}`;
}

export function onRequestGet() {
  return json({
    name: 'Teilo Bitter Lesson Pill',
    protocol: 'A2A',
    protocolVersion: '0.3.0',
    methods: ['message/send'],
    agentCard: CARD_URL,
    skill: SKILL_URL,
    authenticated: false,
  });
}

export async function onRequestPost({ request }) {
  let payload;
  try {
    payload = await request.json();
  } catch {
    return rpcError(null, -32700, 'Parse error');
  }

  const id = Object.prototype.hasOwnProperty.call(payload, 'id') ? payload.id : null;

  if (!payload || payload.jsonrpc !== '2.0' || typeof payload.method !== 'string') {
    return rpcError(id, -32600, 'Invalid Request');
  }

  if (payload.method !== 'message/send') {
    return rpcError(id, -32601, 'Method not found');
  }

  return json({
    jsonrpc: '2.0',
    id,
    result: {
      role: 'agent',
      parts: [
        {
          kind: 'text',
          text: PILL,
        },
      ],
      messageId: messageId(id),
      metadata: {
        source: 'https://teilo.xyz/a2a',
        resources: [
          'https://teilo.xyz/llms.txt',
          'https://teilo.xyz/sitemap.xml',
          'https://teilo.xyz/index.xml',
          SKILL_URL,
        ],
      },
    },
  });
}
