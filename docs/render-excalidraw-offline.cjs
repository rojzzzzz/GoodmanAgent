// Offline renderer for the simple, clean Excalidraw elements used in this diagram.
// Usage: node render-excalidraw-offline.cjs diagram.excalidraw
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawnSync } = require('child_process');
const file = path.resolve(process.argv[2]);
const data = JSON.parse(fs.readFileSync(file, 'utf8'));
const elements = data.elements.filter(e => !e.isDeleted);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
for (const e of elements) {
  const points = ['line', 'arrow'].includes(e.type) && e.points ? e.points.map(([x,y]) => [e.x+x,e.y+y]) : [[e.x,e.y],[e.x+e.width,e.y+e.height]];
  for (const [x,y] of points) { minX=Math.min(minX,x); minY=Math.min(minY,y); maxX=Math.max(maxX,x); maxY=Math.max(maxY,y); }
}
const pad = 40, width = Math.ceil(maxX-minX+pad*2), height = Math.ceil(maxY-minY+pad*2);
const output = [];
for (const e of elements) {
  const stroke = esc(e.strokeColor || '#1e1e1e'), fill = e.backgroundColor && e.backgroundColor !== 'transparent' ? esc(e.backgroundColor) : 'none';
  const common = `stroke="${stroke}" stroke-width="${e.strokeWidth || 1}" opacity="${(e.opacity ?? 100)/100}"`;
  const dash = e.strokeStyle === 'dashed' ? ' stroke-dasharray="8 6"' : e.strokeStyle === 'dotted' ? ' stroke-dasharray="2 5"' : '';
  const rotate = e.angle ? ` transform="rotate(${e.angle*180/Math.PI} ${e.x+e.width/2} ${e.y+e.height/2})"` : '';
  if (e.type === 'rectangle') output.push(`<rect x="${e.x}" y="${e.y}" width="${e.width}" height="${e.height}" rx="${e.roundness ? Math.min(16,e.width/4,e.height/4) : 0}" fill="${fill}" ${common}${dash}${rotate}/>`);
  else if (e.type === 'ellipse') output.push(`<ellipse cx="${e.x+e.width/2}" cy="${e.y+e.height/2}" rx="${e.width/2}" ry="${e.height/2}" fill="${fill}" ${common}${dash}${rotate}/>`);
  else if (e.type === 'diamond') output.push(`<polygon points="${e.x+e.width/2},${e.y} ${e.x+e.width},${e.y+e.height/2} ${e.x+e.width/2},${e.y+e.height} ${e.x},${e.y+e.height/2}" fill="${fill}" ${common}${dash}${rotate}/>`);
  else if (e.type === 'line' || e.type === 'arrow') {
    const p=e.points.map(([x,y])=>[e.x+x,e.y+y]);
    output.push(`<polyline points="${p.map(q=>q.join(',')).join(' ')}" fill="none" ${common}${dash} stroke-linecap="round" stroke-linejoin="round"/>`);
    if (e.type === 'arrow' && e.endArrowhead !== null && p.length > 1) {
      const [x,y]=p[p.length-1], [px,py]=p[p.length-2], a=Math.atan2(y-py,x-px), size=10+Number(e.strokeWidth||1);
      output.push(`<path d="M ${x-size*Math.cos(a-0.5)} ${y-size*Math.sin(a-0.5)} L ${x} ${y} L ${x-size*Math.cos(a+0.5)} ${y-size*Math.sin(a+0.5)}" fill="none" ${common} stroke-linecap="round" stroke-linejoin="round"/>`);
    }
  } else if (e.type === 'text') {
    const size=e.fontSize||20, lineHeight=(e.lineHeight||1.25)*size, anchor=e.textAlign==='center'?'middle':e.textAlign==='right'?'end':'start', x=e.x+(anchor==='middle'?e.width/2:anchor==='end'?e.width:0);
    const lines=(e.text||'').split('\n');
    output.push(`<text x="${x}" y="${e.y}" fill="${stroke}" font-family="${e.fontFamily===3?'monospace':'Arial, sans-serif'}" font-size="${size}" text-anchor="${anchor}" dominant-baseline="text-before-edge" xml:space="preserve" style="white-space:pre"${rotate}>${lines.map((line,i)=>`<tspan x="${x}" dy="${i?lineHeight:0}">${esc(line)||'&#160;'}</tspan>`).join('')}</text>`);
  } else console.warn('Unsupported element:', e.type, e.id);
}
const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="${minX-pad} ${minY-pad} ${width} ${height}"><rect x="${minX-pad}" y="${minY-pad}" width="${width}" height="${height}" fill="${esc(data.appState?.viewBackgroundColor || '#ffffff')}"/>${output.join('')}</svg>`;
const svgPath=file.replace(/\.excalidraw$/i,'.svg'), pngPath=file.replace(/\.excalidraw$/i,'.png');
fs.writeFileSync(svgPath,svg);
// Render local SVG with installed Chrome; no npm packages or remote imports.
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'goodman-diagram-render-'));
try {
  const result = spawnSync('C:/Program Files/Google/Chrome/Application/chrome.exe', [
    '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run',
    '--no-default-browser-check', '--disable-background-networking',
    '--force-device-scale-factor=1', `--user-data-dir=${profile}`,
    `--window-size=${width},${height}`, `--screenshot=${pngPath}`,
    'file:///' + svgPath.replace(/\\/g, '/'),
  ], { windowsHide: true, encoding: 'utf8', timeout: 30000 });
  if (result.error || result.status !== 0 || !fs.existsSync(pngPath)) {
    throw result.error || new Error(result.stderr || 'Chrome did not render a PNG');
  }
  console.log(`Rendered ${svgPath} and ${pngPath} (${width} × ${height})`);
} finally {
  // Only remove the unique temporary profile created by this process.
  const target = path.resolve(profile);
  const tempRoot = path.resolve(os.tmpdir()) + path.sep;
  if (!target.startsWith(tempRoot) || !path.basename(target).startsWith('goodman-diagram-render-')) {
    throw new Error('Unexpected temporary profile path');
  }
  fs.rmSync(target, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
}
