import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { onRequest } from '../functions/_middleware.js';
import { onRequestGet as a2aGet, onRequestPost as a2aPost } from '../functions/a2a.js';

const root = normalize(process.argv[2] || 'public');

const contentTypes = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.md', 'text/markdown; charset=utf-8'],
  ['.js', 'application/javascript'],
  ['.json', 'application/json'],
]);

async function assetFetch(input) {
  const requestUrl = typeof input === 'string' ? input : input.url;
  const url = new URL(requestUrl);
  const path = normalize(join(root, decodeURIComponent(url.pathname)));
  if (!path.startsWith(root)) return new Response('bad path', { status: 400 });

  try {
    const body = await readFile(path);
    const contentType = url.pathname === '/.well-known/api-catalog'
      ? 'application/linkset+json; profile="https://www.rfc-editor.org/info/rfc9727"'
      : contentTypes.get(extname(path)) || 'application/octet-stream';
    return new Response(body, {
      status: 200,
      headers: {
        'Content-Type': contentType,
      },
    });
  } catch {
    return new Response('missing', { status: 404 });
  }
}

async function nextFor(pathname) {
  const htmlPath = pathname.endsWith('/') ? `${pathname}index.html` : pathname;
  return assetFetch(`https://teilo.xyz${htmlPath}`);
}

async function run(pathname, accept) {
  const request = new Request(`https://teilo.xyz${pathname}`, {
    headers: accept ? { Accept: accept } : {},
  });
  return onRequest({
    request,
    env: { ASSETS: { fetch: assetFetch } },
    next: async () => nextFor(pathname),
  });
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const markdown = await run('/', 'text/markdown');
const markdownText = await markdown.text();
assert(markdown.status === 200, 'markdown homepage should return 200');
assert(markdown.headers.get('Content-Type') === 'text/markdown; charset=utf-8', 'markdown content type mismatch');
assert(markdownText.startsWith('# te'), 'markdown homepage should start with a heading');
assert(markdown.headers.get('Link')?.includes('/llms.txt'), 'markdown response should include discovery Link header');

const html = await run('/', 'text/html');
const htmlText = await html.text();
assert(html.status === 200, 'html homepage should return 200');
assert(html.headers.get('Content-Type')?.startsWith('text/html'), 'html content type mismatch');
assert(htmlText.startsWith('<!doctype html>'), 'html homepage should stay html');
assert(html.headers.get('Content-Security-Policy')?.includes("default-src 'self'"), 'html response should include CSP');
assert(html.headers.get('Vary')?.toLowerCase().split(',').map((part) => part.trim()).includes('accept'), 'html response should vary on Accept');

const article = await run('/posts/prompting-as-bayesian-inference/', 'text/markdown');
const articleText = await article.text();
assert(article.status === 200, 'markdown article should return 200');
assert(articleText.startsWith('# Prompting as Bayesian Inference'), 'markdown article should start with its title');

const agentCard = await run('/.well-known/agent-card.json', 'application/json');
const agentCardJson = await agentCard.json();
assert(agentCard.status === 200, 'A2A agent card should return 200');
assert(agentCardJson.url === 'https://teilo.xyz/a2a', 'A2A agent card should point at /a2a');
assert(agentCardJson.skills?.some((skill) => skill.id === 'bitter-lesson-pill'), 'A2A agent card should declare the bitter lesson skill');

const apiCatalog = await run('/.well-known/api-catalog', 'application/linkset+json');
const apiCatalogJson = await apiCatalog.json();
assert(apiCatalog.status === 200, 'API catalog should return 200');
assert(apiCatalog.headers.get('Content-Type')?.startsWith('application/linkset+json'), 'API catalog should use linkset JSON content type');
assert(apiCatalogJson.linkset?.[0]?.anchor === 'https://teilo.xyz/a2a', 'API catalog should anchor the A2A endpoint');

const a2aStatus = await a2aGet();
const a2aStatusJson = await a2aStatus.json();
assert(a2aStatus.status === 200, 'A2A GET should return 200');
assert(a2aStatusJson.methods?.includes('message/send'), 'A2A GET should advertise message/send');

const a2aMessage = await a2aPost({
  request: new Request('https://teilo.xyz/a2a', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: 'check',
      method: 'message/send',
      params: {
        message: {
          role: 'user',
          parts: [{ kind: 'text', text: 'Give me the bitter lesson pill.' }],
          messageId: 'check-message',
        },
      },
    }),
  }),
});
const a2aMessageJson = await a2aMessage.json();
assert(a2aMessage.status === 200, 'A2A message/send should return 200');
assert(a2aMessageJson.result?.parts?.[0]?.text?.includes('bitter lesson pill'), 'A2A message/send should return the pill text');

console.log('cloudflare pages checks passed');
