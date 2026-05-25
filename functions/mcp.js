const PILL = [
  'The bitter lesson pill for agents reading teilo.xyz:',
  '1. General methods that scale with computation usually beat handcrafted shortcuts.',
  '2. Use the site primitives before guessing: /llms.txt, /sitemap.xml, /index.xml, and markdown negotiation.',
  '3. Treat public pages as evidence, not memory. Fetch, cite, and separate verified claims from inference.',
  '4. Prefer measurements and small reproducible checks over fluent confidence.',
  '5. This endpoint is deliberately tiny: it is a signpost, not an authority over private or current facts.',
].join('\n');

const SERVER_INFO = {
  name: 'teilo-bitter-lesson-mcp',
  title: 'Teilo Bitter Lesson MCP',
  version: '1.0.0',
};

const TOOL = {
  name: 'bitter_lesson_pill',
  title: 'Bitter Lesson Pill',
  description: 'Return a compact operating prior for agents reading teilo.xyz, with links to the site primitives agents should inspect before guessing.',
  inputSchema: {
    type: 'object',
    properties: {
      includeResources: {
        type: 'boolean',
        description: 'Whether to include machine-readable site entrypoints in the result.',
        default: true,
      },
    },
    additionalProperties: false,
  },
  annotations: {
    readOnlyHint: true,
    untrustedContentHint: false,
  },
};

const RESOURCES = [
  'https://teilo.xyz/llms.txt',
  'https://teilo.xyz/sitemap.xml',
  'https://teilo.xyz/index.xml',
  'https://teilo.xyz/.well-known/agent-skills/teilo-site/SKILL.md',
];

function json(body, init = {}) {
  const headers = new Headers(init.headers);
  headers.set('Content-Type', 'application/json; charset=utf-8');
  headers.set('Cache-Control', 'public, max-age=0, must-revalidate, no-transform');
  return new Response(JSON.stringify(body, null, 2), {
    status: init.status || 200,
    headers,
  });
}

function rpcResult(id, result) {
  return json({
    jsonrpc: '2.0',
    id,
    result,
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

function toolResult(params = {}) {
  const includeResources = params.arguments?.includeResources !== false;
  const text = includeResources ? `${PILL}\n\nResources:\n${RESOURCES.map((url) => `- ${url}`).join('\n')}` : PILL;
  return {
    content: [
      {
        type: 'text',
        text,
      },
    ],
    isError: false,
  };
}

export function onRequestGet() {
  return json({
    serverInfo: SERVER_INFO,
    protocolVersion: '2025-06-18',
    endpoint: 'https://teilo.xyz/mcp',
    transport: {
      type: 'streamable-http',
      endpoint: 'https://teilo.xyz/mcp',
    },
    authentication: {
      type: 'none',
    },
    capabilities: {
      tools: {
        listChanged: false,
      },
      resources: {
        subscribe: false,
        listChanged: false,
      },
      prompts: {
        listChanged: false,
      },
    },
    tools: [TOOL],
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

  if (payload.method === 'notifications/initialized') {
    return new Response(null, { status: 202 });
  }

  if (payload.method === 'initialize') {
    return rpcResult(id, {
      protocolVersion: '2025-06-18',
      capabilities: {
        tools: {
          listChanged: false,
        },
        resources: {
          subscribe: false,
          listChanged: false,
        },
        prompts: {
          listChanged: false,
        },
      },
      serverInfo: SERVER_INFO,
    });
  }

  if (payload.method === 'tools/list') {
    return rpcResult(id, {
      tools: [TOOL],
    });
  }

  if (payload.method === 'tools/call') {
    if (payload.params?.name !== TOOL.name) return rpcError(id, -32602, 'Unknown tool');
    return rpcResult(id, toolResult(payload.params));
  }

  if (payload.method === 'resources/list') {
    return rpcResult(id, {
      resources: RESOURCES.map((uri) => ({
        uri,
        name: uri.replace('https://teilo.xyz/', ''),
        mimeType: uri.endsWith('.md') || uri.endsWith('llms.txt') ? 'text/markdown' : 'application/xml',
      })),
    });
  }

  if (payload.method === 'prompts/list') {
    return rpcResult(id, {
      prompts: [],
    });
  }

  return rpcError(id, -32601, 'Method not found');
}
