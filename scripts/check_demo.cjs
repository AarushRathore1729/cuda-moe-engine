// Exercise the offline explorer against its embedded raw samples without a browser.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'demo/index.html'), 'utf8');
const raw = html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code = html.match(/<\/script><script>([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(raw);
assert.deepStrictEqual(data, JSON.parse(fs.readFileSync(path.join(root,
  'benchmarks/a100-block/block_results.json'), 'utf8')));
const elements = {};
for (const id of ['data', 'tokens', 'pattern', 'stage', 'statistic', 'cases', 'evidence', 'download']) {
  elements[id] = {value:'all', innerHTML:'', textContent:'', events:{},
    addEventListener(event, callback){this.events[event] = callback;}};
}
elements.data.textContent = raw;
elements.stage.value = 'forward';
elements.statistic.value = 'median_us';
let downloadedBlob, anchor;
const context = vm.createContext({
  document:{getElementById:id=>elements[id], createElement:()=>{
    anchor = {clicked:false, click(){this.clicked=true;}}; return anchor;
  }},
  Blob,
  URL:{createObjectURL:b=>{downloadedBlob=b;return 'blob:test';}, revokeObjectURL(){}},
  setTimeout:cb=>cb(),
});
vm.runInContext(code, context);
assert.strictEqual((elements.cases.innerHTML.match(/<section/g)||[]).length, 8);
let checks = 0;
for (const tokens of ['all', '128', '2048']) for (const pattern of ['all', 'balanced', 'skewed']) {
  for (const stage of ['forward', 'router', 'dispatch', 'experts', 'combine']) {
    for (const stat of ['median_us', 'p95_us']) {
      Object.assign(elements.tokens,{value:tokens}); Object.assign(elements.pattern,{value:pattern});
      Object.assign(elements.stage,{value:stage}); Object.assign(elements.statistic,{value:stat});
      elements.stage.events.change();
      const cases=data.configurations.filter(c=>(tokens==='all'||String(c.tokens)===tokens)&&
        (pattern==='all'||c.pattern===pattern));
      assert.strictEqual((elements.cases.innerHTML.match(/<section/g)||[]).length,cases.length);
      const ratios=[...elements.cases.innerHTML.matchAll(/class="badge">([\d.]+)×/g)].map(m=>m[1]);
      cases.forEach((c,i)=>{
        const get=impl=>data.benchmarks.find(r=>r.case_id===c.case_id&&r.stage===stage&&r.implementation===impl)[stat];
        assert.strictEqual(ratios[i],(get('torch')/get('cuda_fused')).toFixed(2));
      });
      assert(!elements.cases.innerHTML.includes('NaN'));
      checks++;
    }
  }
}
elements.download.events.click();
assert(anchor.clicked && anchor.download==='a100_block_results.json');
downloadedBlob.text().then(text=>{
  assert.deepStrictEqual(JSON.parse(text),data);
  console.log(`Explorer verified: ${checks} filter/stage/statistic combinations; JSON download matches.`);
}).catch(error=>{console.error(error);process.exitCode=1;});
