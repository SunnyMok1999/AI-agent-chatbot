import fs from 'fs';
import path from 'path';
import axios from 'axios';
import * as cheerio from 'cheerio';

type CliArgs = {
  urlsFile: string;
  outDir: string;
  manifestPath: string;
  manifestId: string;
  timeoutMs: number;
};

type ManifestRow = {
  manifest_id?: string;
  local_path?: string;
  local_paths?: string[];
  [key: string]: unknown;
};

const defaultArgs: CliArgs = {
  urlsFile: path.resolve('data/seed/feynman_004_khan_urls.txt'),
  outDir: path.resolve('data/seed/feynman_004_khan_md'),
  manifestPath: path.resolve('data/seed/ingest_manifest.json'),
  manifestId: 'feynman_004',
  timeoutMs: 30000,
};

function printUsage() {
  console.log('Usage: npm run khan:convert -- [--urls <file>] [--out <dir>] [--manifest <file>] [--id <manifest_id>]');
  console.log('Example: npm run khan:convert -- --urls data/seed/feynman_004_khan_urls.txt --id feynman_004');
}

function parseArgs(argv: string[]): CliArgs {
  const args = { ...defaultArgs };

  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];

    if (token === '--help' || token === '-h') {
      printUsage();
      process.exit(0);
    }

    if (token === '--urls' && argv[i + 1]) {
      args.urlsFile = path.resolve(argv[++i]);
      continue;
    }

    if (token === '--out' && argv[i + 1]) {
      args.outDir = path.resolve(argv[++i]);
      continue;
    }

    if (token === '--manifest' && argv[i + 1]) {
      args.manifestPath = path.resolve(argv[++i]);
      continue;
    }

    if (token === '--id' && argv[i + 1]) {
      args.manifestId = argv[++i];
      continue;
    }

    if (token === '--timeout' && argv[i + 1]) {
      const n = Number(argv[++i]);
      if (!Number.isNaN(n) && n > 0) args.timeoutMs = n;
      continue;
    }
  }

  return args;
}

function ensureDir(dir: string) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function readUrlList(filePath: string): string[] {
  if (!fs.existsSync(filePath)) {
    throw new Error(`URLs file not found: ${filePath}`);
  }

  return fs
    .readFileSync(filePath, 'utf-8')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith('#'));
}

function slugify(input: string): string {
  return input
    .toLowerCase()
    .replace(/https?:\/\//g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 120);
}

function normalizeText(text: string): string {
  return text
    .replace(/\u00a0/g, ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function extractMainText(html: string): { title: string; content: string } {
  const $ = cheerio.load(html);

  $('script,style,noscript,svg,header,footer,nav,form').remove();

  const title = normalizeText($('title').first().text() || 'Khan Academy Content');

  const selectors = [
    '[data-test-id="video-transcript"]',
    '[class*="transcript"]',
    'main',
    'article',
    '[role="main"]',
    '.article-content',
    '.khan-article-content',
    '.framework-content',
    '.perseus-renderer',
    '.content',
  ];

  let best = '';
  for (const selector of selectors) {
    const text = normalizeText($(selector).first().text());
    if (text.length > best.length) {
      best = text;
    }
  }

  if (!best) {
    best = normalizeText($('body').text());
  }

  return { title, content: best };
}

async function fetchPayload(url: string, timeoutMs: number): Promise<string> {
  const fetchReadableFallback = () => fetchReadablePayload(url, timeoutMs);

  try {
    const response = await axios.get<string>(url, {
      timeout: timeoutMs,
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        Referer: 'https://www.khanacademy.org/',
      },
    });

    const direct = response.data || '';

    // Khan pages can return tiny JS bootstrap HTML to non-browser clients.
    // In that case, use the readable fallback instead.
    if (direct.length < 5000 || /trouble loading external resources/i.test(direct)) {
      return fetchReadableFallback();
    }

    return direct;
  } catch (error) {
    if (!axios.isAxiosError(error) || !error.response || (error.response.status !== 403 && error.response.status !== 429)) {
      throw error;
    }

    return fetchReadableFallback();
  }
}

async function fetchReadablePayload(url: string, timeoutMs: number): Promise<string> {
  const fallbackUrl = `https://r.jina.ai/http://${url.replace(/^https?:\/\//, '')}`;
  const fallback = await axios.get<string>(fallbackUrl, {
    timeout: timeoutMs,
    headers: {
      'User-Agent': 'Mozilla/5.0',
    },
  });

  return fallback.data;
}

function extractTextFromUnknownPayload(payload: string): { title: string; content: string } {
  const looksLikeHtml = /<html|<body|<main|<article/i.test(payload);
  if (looksLikeHtml) {
    return extractMainText(payload);
  }

  const lines = payload.split(/\r?\n/).map((x) => x.trim());
  const firstNonEmpty = lines.find((x) => x.length > 0) || 'Khan Academy Content';
  return {
    title: normalizeText(firstNonEmpty.replace(/^#\s+/, '')),
    content: normalizeText(payload),
  };
}

function buildMarkdown(url: string, title: string, content: string): string {
  return [
    `# ${title || 'Khan Academy Content'}`,
    '',
    `Source: ${url}`,
    '',
    content,
    '',
  ].join('\n');
}

function updateManifest(manifestPath: string, manifestId: string, localPaths: string[]) {
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`Manifest file not found: ${manifestPath}`);
  }

  const parsed = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as ManifestRow[];
  if (!Array.isArray(parsed)) {
    throw new Error('Manifest JSON must be an array');
  }

  const idx = parsed.findIndex((row) => row.manifest_id === manifestId);
  if (idx < 0) {
    throw new Error(`manifest_id not found: ${manifestId}`);
  }

  parsed[idx].local_path = '';
  parsed[idx].local_paths = localPaths;

  fs.writeFileSync(manifestPath, `${JSON.stringify(parsed, null, 2)}\n`, 'utf-8');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  ensureDir(args.outDir);

  const urls = readUrlList(args.urlsFile);
  if (!urls.length) {
    console.log('No URLs found. Add at least one URL to your list file.');
    return;
  }

  console.log(`Found ${urls.length} Khan Academy URLs`);

  const outputFiles: string[] = [];

  for (let i = 0; i < urls.length; i++) {
    const url = urls[i];
    const fileName = `${String(i + 1).padStart(2, '0')}-${slugify(url)}.md`;
    const outPath = path.resolve(args.outDir, fileName);

    try {
      console.log(`Fetching: ${url}`);
      const payload = await fetchPayload(url, args.timeoutMs);
      let { title, content } = extractTextFromUnknownPayload(payload);

      if ((!content || content.length < 100) && /<html|<body|<main|<article/i.test(payload)) {
        const readablePayload = await fetchReadablePayload(url, args.timeoutMs);
        const retried = extractTextFromUnknownPayload(readablePayload);
        title = retried.title;
        content = retried.content;
      }

      if (!content || content.length < 100) {
        console.log(`⚠️ Skipped (not enough extractable text): ${url}`);
        continue;
      }

      const md = buildMarkdown(url, title, content);
      fs.writeFileSync(outPath, md, 'utf-8');
      outputFiles.push(outPath);
      console.log(`✅ Saved ${outPath}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.log(`❌ Failed ${url}: ${message}`);
    }
  }

  if (!outputFiles.length) {
    console.log('No markdown files generated.');
    return;
  }

  updateManifest(args.manifestPath, args.manifestId, outputFiles);

  console.log(`\nUpdated ${args.manifestId} in manifest with ${outputFiles.length} local_paths.`);
  console.log(`Manifest: ${args.manifestPath}`);
  console.log(`Output dir: ${args.outDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
