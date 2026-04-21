import { NextResponse } from 'next/server';

const backendOrigin = (process.env.BACKEND_ORIGIN || 'http://127.0.0.1:3004').replace(/\/$/, '');

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type DuplexRequestInit = RequestInit & {
  duplex?: 'half';
};

async function forwardRequest(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const incomingUrl = new URL(request.url);
  const targetUrl = `${backendOrigin}/api/${path.join('/')}${incomingUrl.search}`;

  const headers = new Headers(request.headers);
  headers.delete('host');
  headers.delete('connection');
  headers.delete('content-length');

  try {
    const hasRequestBody = request.method !== 'GET' && request.method !== 'HEAD';
    const fetchOptions: DuplexRequestInit = {
      method: request.method,
      headers,
      body: hasRequestBody ? request.body : undefined,
      duplex: hasRequestBody ? 'half' : undefined,
      redirect: 'manual',
    };
    const upstreamResponse = await fetch(targetUrl, fetchOptions);
    const responseBody = await upstreamResponse.arrayBuffer();

    const responseHeaders = new Headers(upstreamResponse.headers);
    responseHeaders.delete('content-length');
    responseHeaders.delete('content-encoding');
    responseHeaders.delete('transfer-encoding');

    return new Response(responseBody, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    const detail =
      error instanceof Error
        ? `Backend service is unavailable at ${backendOrigin}. ${error.message}`
        : `Backend service is unavailable at ${backendOrigin}.`;

    return NextResponse.json(
      { detail },
      { status: 503 },
    );
  }
}

export async function GET(request: Request, context: { params: Promise<{ path: string[] }> }) {
  return forwardRequest(request, context);
}

export async function POST(request: Request, context: { params: Promise<{ path: string[] }> }) {
  return forwardRequest(request, context);
}

export async function PUT(request: Request, context: { params: Promise<{ path: string[] }> }) {
  return forwardRequest(request, context);
}

export async function PATCH(request: Request, context: { params: Promise<{ path: string[] }> }) {
  return forwardRequest(request, context);
}

export async function DELETE(request: Request, context: { params: Promise<{ path: string[] }> }) {
  return forwardRequest(request, context);
}

export async function OPTIONS(request: Request, context: { params: Promise<{ path: string[] }> }) {
  return forwardRequest(request, context);
}

export async function HEAD(request: Request, context: { params: Promise<{ path: string[] }> }) {
  return forwardRequest(request, context);
}
