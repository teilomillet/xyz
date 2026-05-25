const SECURITY_HEADERS = {
  'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
  'X-Frame-Options': 'DENY',
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Permissions-Policy': 'accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()',
  'Content-Security-Policy': "default-src 'self'; base-uri 'self'; object-src 'none'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https:; font-src 'self' https://cdn.jsdelivr.net; connect-src 'self'; frame-ancestors 'none'; form-action 'self'; upgrade-insecure-requests",
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Resource-Policy': 'same-origin',
  Link: '</llms.txt>; rel="describedby"; type="text/plain", </sitemap.xml>; rel="sitemap"; type="application/xml", </.well-known/agent-skills/index.json>; rel="describedby"; type="application/json"',
};

function wantsMarkdown(request) {
  const accept = request.headers.get('Accept') || '';
  return /\btext\/markdown\b/i.test(accept);
}

function markdownPath(pathname) {
  if (pathname === '/') return '/index.md';
  if (pathname.endsWith('/')) return `${pathname}index.md`;
  if (!pathname.split('/').pop().includes('.')) return `${pathname}/index.md`;
  return null;
}

function withDocumentHeaders(response, contentType) {
  const headers = new Headers(response.headers);
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) {
    headers.set(name, value);
  }
  headers.set('Vary', appendVary(headers.get('Vary'), 'Accept'));
  headers.set('Cache-Control', 'public, max-age=0, must-revalidate');
  if (contentType) headers.set('Content-Type', contentType);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function appendVary(current, token) {
  if (!current) return token;
  const parts = current.split(',').map((part) => part.trim().toLowerCase());
  return parts.includes(token.toLowerCase()) ? current : `${current}, ${token}`;
}

export async function onRequest(context) {
  if ((context.request.method === 'GET' || context.request.method === 'HEAD') && wantsMarkdown(context.request)) {
    const url = new URL(context.request.url);
    const mdPath = markdownPath(url.pathname);
    if (mdPath) {
      const assetUrl = new URL(mdPath, url.origin);
      const markdown = await context.env.ASSETS.fetch(assetUrl.toString());
      if (markdown.ok) {
        return withDocumentHeaders(markdown, 'text/markdown; charset=utf-8');
      }
    }
  }

  const response = await context.next();
  return withDocumentHeaders(response);
}
