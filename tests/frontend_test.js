// 前端撤回逻辑功能测试：DOM 打桩 + 可控 fetch，验证
//  1) 求解期间修改输入 → 旧建议立即撤下（结果区清空）
//  2) 迟到的旧响应即使成功返回也绝不恢复显示
//  3) 正常求解 → 结果正常渲染
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'app', 'static', 'index.html'), 'utf8');
const pageJs = html.match(/<script>([\s\S]*)<\/script>/)[1];

// ---- DOM 打桩 ----
const store = {};
function el(id) {
  if (!store[id]) {
    const e = {
      id, checked: false, textContent: '', innerHTML: '',
      style: { display: 'none' }, dataset: {}, disabled: false,
      listeners: {},
      classList: { add() {}, remove() {} },
      addEventListener(ev, fn) { (this.listeners[ev] ??= []).push(fn); },
      appendChild() {},
    };
    let v = '0';
    Object.defineProperty(e, 'value', {   // 真实 DOM 中 input.value 恒为字符串
      get: () => v, set: x => { v = String(x); },
    });
    store[id] = e;
  }
  return store[id];
}
global.document = {
  getElementById: el,
  querySelector: () => el('tbody'),
  createElement: () => el('tr_' + Math.random()),
};
let fetchImpl = () => Promise.reject(new Error('fetch not stubbed'));
global.fetch = (...a) => fetchImpl(...a);

const META = { feature_names: ['A','B','C','D'], total_grams: 1000, proba_threshold: 0.7,
  n_trees: 5, max_depth: 3, tree_depths: [3,3,3,3,3], train_seed: 1 };
const SAMPLES = { feasible: { original: [250,250,350,150], lower: [0,0,0,0],
  upper: [1000,1000,1000,1000], locked: [false,true,false,false] },
  infeasible: { original: [150,130,620,100], lower: [0,0,620,0],
  upper: [150,1000,1000,1000], locked: [false,false,false,false] } };
const OK_RESULT = { status: 'ok', suggestion: [390,250,210,150], probability: 0.7969,
  changed_count: 2, total_change: 280, solve_ms: 50, optimality: '已证明最优…',
  per_ingredient: [
    { name:'A', original:250, new:390, delta:140, locked:false },
    { name:'B', original:250, new:250, delta:0, locked:true },
    { name:'C', original:350, new:210, delta:-140, locked:false },
    { name:'D', original:150, new:150, delta:0, locked:false } ] };

// boot 在 eval 时同步发起 fetch，须提前打桩
fetchImpl = url => Promise.resolve({ json: async () => url.includes('meta') ? META : SAMPLES });

const driver = `
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const assert = (name, cond) => { console.log((cond ? '[PASS] ' : '[FAIL] ') + name); if (!cond) process.exitCode = 1; };

  await sleep(20);
  assert('启动后载入可行样例', document.getElementById('orig0').value == 250);

  // --- 场景 1：正常求解渲染 ---
  fetchImpl = () => Promise.resolve({ json: async () => OK_RESULT });
  await solve();
  assert('正常求解后 ok 面板显示', document.getElementById('panelOk').style.display === 'block');
  assert('概率已渲染', document.getElementById('okProb').textContent.includes('79.69'));

  // --- 场景 2：求解期间修改 → 立即撤下；迟到结果不恢复 ---
  let resolveStale;
  fetchImpl = () => new Promise(res => { resolveStale = () => res({ json: async () => OK_RESULT }); });
  const p = solve();                 // 发起求解（不等待）
  await sleep(5);
  onUserEdit();                      // 用户修改输入
  assert('修改后旧建议立即撤下', document.getElementById('panelOk').style.display === 'none');
  assert('状态提示已撤下', document.getElementById('status').textContent.includes('撤下'));
  resolveStale();                    // 迟到响应“绕过”abort 成功返回（竞态）
  await p;
  await sleep(5);
  assert('迟到结果未恢复 ok 面板', document.getElementById('panelOk').style.display === 'none');
  assert('迟到结果未改写状态', document.getElementById('status').textContent.includes('撤下'));

  // --- 场景 3：撤下后重新求解仍正常 ---
  fetchImpl = () => Promise.resolve({ json: async () => OK_RESULT });
  await solve();
  assert('重新求解后 ok 面板恢复', document.getElementById('panelOk').style.display === 'block');

  // --- 场景 4：无解分支渲染 ---
  fetchImpl = () => Promise.resolve({ json: async () => ({ status: 'infeasible', reason: '已证明无解', solve_ms: 30 }) });
  await solve();
  assert('无解面板显示', document.getElementById('panelInfeasible').style.display === 'block');
  assert('ok 面板保持隐藏', document.getElementById('panelOk').style.display === 'none');
})();
`;

eval(pageJs + driver);
