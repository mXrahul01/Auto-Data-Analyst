export type Service = { name: string; state: 'reachable' | 'unavailable'; stage: string; health_path: string };
export type Catalog = { stage: string; services: Service[] };

export async function getCatalog(signal?: AbortSignal): Promise<Catalog> {
  const response = await fetch('/api/v1/services', { signal });
  if (!response.ok) throw new Error('Service status could not be loaded. Check the Compose stack.');
  const body: unknown = await response.json();
  if (!body || typeof body !== 'object' || !('services' in body) || !Array.isArray(body.services)) {
    throw new Error('The gateway returned an invalid service catalog.');
  }
  const services = body.services;
  for (const item of services) {
    if (!item || typeof item !== 'object' || typeof item.name !== 'string' ||
        !['reachable', 'unavailable'].includes(item.state) || typeof item.stage !== 'string' ||
        typeof item.health_path !== 'string') throw new Error('The service catalog format is invalid.');
  }
  return { stage: 'scaffold', services };
}
