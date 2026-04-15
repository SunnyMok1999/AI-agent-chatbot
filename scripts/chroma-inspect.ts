import dotenv from 'dotenv';

dotenv.config();

const chromaUrl = process.env.CHROMA_URL || 'http://localhost:8001';
const tenant = 'default_tenant';
const database = 'default_database';
const targetCollection = process.env.CHROMA_COLLECTION_NAME || 'ml_documents';

async function request(path: string, init?: RequestInit) {
  const res = await fetch(`${chromaUrl}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
    ...init,
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${await res.text()}`);
  }

  return res.json();
}

async function main() {
  console.log(`🔎 Chroma URL: ${chromaUrl}`);
  const collections = await request(`/api/v2/tenants/${tenant}/databases/${database}/collections`);

  console.log('\n📚 Collections:');
  for (const c of collections) {
    console.log(`- ${c.name} (${c.id})`);
  }

  const found = collections.find((c: any) => c.name === targetCollection);
  if (!found) {
    console.log(`\n⚠️ Target collection not found: ${targetCollection}`);
    return;
  }

  const count = await request(
    `/api/v2/tenants/${tenant}/databases/${database}/collections/${found.id}/count`,
  );
  console.log(`\n📦 ${targetCollection} count: ${count}`);

  const sample = await request(
    `/api/v2/tenants/${tenant}/databases/${database}/collections/${found.id}/get`,
    {
      method: 'POST',
      body: JSON.stringify({
        limit: 3,
        offset: 0,
        include: ['documents', 'metadatas'],
      }),
    },
  );

  console.log('\n🧪 Sample documents:');
  const docs = sample.documents || [];
  const metas = sample.metadatas || [];
  for (let i = 0; i < docs.length; i++) {
    const source = metas[i]?.source || 'unknown-source';
    const preview = String(docs[i] || '').replace(/\s+/g, ' ').slice(0, 140);
    console.log(`- [${i + 1}] source=${source}`);
    console.log(`  ${preview}${preview.length >= 140 ? '...' : ''}`);
  }
}

main().catch((err) => {
  console.error('❌ Failed to inspect Chroma:', err?.message || err);
  process.exit(1);
});
