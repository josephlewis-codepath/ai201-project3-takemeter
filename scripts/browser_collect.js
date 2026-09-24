/* TakeMeter — browser collector (Route B)
 *
 * Reddit blocks scripts but not your logged-in browser. This runs in the page's
 * own session and downloads the raw JSON, which scripts/collect_reddit.py then
 * turns into the labeled-CSV skeleton.
 *
 * HOW TO RUN
 *   1. Open https://www.reddit.com/r/fantasyfootball/ in Chrome, logged in.
 *   2. View > Developer > JavaScript Console   (or Cmd+Option+J)
 *   3. If it says "Allow pasting", type: allow pasting  then Enter.
 *   4. Paste this whole file, press Enter, and leave the tab in the foreground.
 *   5. Chrome will ask to "allow multiple downloads" -> Allow.
 *   6. Takes ~2 minutes. Five files land in ~/Downloads, named ff_*.json.
 *
 * THEN
 *   mkdir -p ~/ai201-project3-takemeter/saved
 *   mv ~/Downloads/ff_*.json ~/ai201-project3-takemeter/saved/
 *   cd ~/ai201-project3-takemeter
 *   python3 scripts/collect_reddit.py --files "saved/*.json" --out data/raw_unlabeled.csv
 */

(async () => {
  const SUB = 'fantasyfootball';
  const PAUSE = 1200;               // be polite; this is someone else's server
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  const get = async path => {
    try {
      const r = await fetch(path, { credentials: 'include', headers: { Accept: 'application/json' } });
      if (!r.ok) { console.warn('  HTTP', r.status, path); return null; }
      return await r.json();
    } catch (e) {
      console.warn('  error', path, e.message);
      return null;
    }
  };

  const save = (name, payloads) => {
    const blob = new Blob([JSON.stringify(payloads)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    console.log(`SAVED ${name} (${payloads.length} payloads)`);
  };

  // --- 1. sub-wide recent comment stream -----------------------------------
  console.log('--- recent comment stream ---');
  const stream = [];
  let after = null;
  for (let i = 0; i < 4; i++) {
    const d = await get(`/r/${SUB}/comments.json?limit=100${after ? '&after=' + after : ''}`);
    if (!d) break;
    stream.push(d);
    console.log(`  page ${i + 1} ok`);
    after = d?.data?.after;
    await sleep(PAUSE);
    if (!after) break;
  }
  if (stream.length) save('ff_recent_comments.json', stream);

  // --- 2. listings, each with its own threads ------------------------------
  const SOURCES = [
    ['ff_top_week.json',  'top.json?t=week&limit=100',  14],
    ['ff_top_month.json', 'top.json?t=month&limit=100', 12],
    ['ff_hot.json',       'hot.json?limit=100',         12],
    ['ff_new.json',       'new.json?limit=100',          8],
  ];

  for (const [filename, endpoint, nThreads] of SOURCES) {
    console.log(`--- ${filename} ---`);
    const bucket = [];
    const d = await get(`/r/${SUB}/${endpoint}`);
    if (!d) { console.warn('  listing failed, skipping'); continue; }
    bucket.push(d);
    await sleep(PAUSE);

    const posts = (d?.data?.children || [])
      .map(c => c.data)
      .filter(p => p && !p.stickied && p.permalink);

    for (const p of posts.slice(0, nThreads)) {
      const t = await get(p.permalink.replace(/\/$/, '') + '.json?limit=200&sort=top');
      if (t) { bucket.push(t); console.log(`  + ${p.permalink.slice(0, 64)}`); }
      await sleep(PAUSE);
    }
    save(filename, bucket);
  }

  console.log('%cDONE — check ~/Downloads for ff_*.json', 'font-weight:bold');
})();
