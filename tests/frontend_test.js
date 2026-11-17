// 前端撤回逻辑功能测试：DOM 打桩 + 可控 fetch，验证
//  1) 求解期间修改输入 → 旧建议立即撤下（结果区清空）
//  2) 迟到的旧响应即使成功返回也绝不恢复显示
//  3) 正常求解 → 结果正常渲染
//  4) 新增批次核查：编辑即撤下、迟到不恢复、三种结论分支与输入错误渲染
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
  n_trees: 5, max_depth: 3, tree_depths: [3,3,3,3,3], train_seed: 1,
  n_reference: 1000, reference_seed: 2, n_permutations: 999, drift_alpha: 0.05,
  drift_min_rows: 20, drift_max_rows: 200 };
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

  // ============ 新增批次适用性核查（独立 epoch，与求解区互不影响） ============
  const REVIEW = { status: 'review', statistic: 0.4123, p_value: 0.001, n_rows: 40,
    n_reference: 1000, n_permutations: 999, elapsed_ms: 190,
    worst_leaf: { tree: 2, leaf_id: 14, batch_proportion: 0.35, reference_proportion: 0.12, abs_diff: 0.23 } };
  const NO_SHIFT = { status: 'no_shift', statistic: 0.08, p_value: 0.62, n_rows: 30,
    n_reference: 1000, n_permutations: 999, elapsed_ms: 180,
    worst_leaf: { tree: 1, leaf_id: 7, batch_proportion: 0.2, reference_proportion: 0.18, abs_diff: 0.02 } };
  const INVALID = { status: 'invalid', errors: ['第 1 行应有 4 个整数克数，实际 3 个。', '第 5 行总重为 999 克，须等于 1000 克。'] };

  // 场景 5：偏移批次 → 人工复核面板，展示差异最大的树/叶节点与两组占比
  fetchImpl = () => Promise.resolve({ json: async () => REVIEW });
  await runDrift();
  assert('偏移批次显示人工复核面板', document.getElementById('panelReview').style.display === 'block');
  assert('复核面板含 p 值', document.getElementById('reviewP').textContent.includes('0.001'));
  const worstHtml = document.getElementById('reviewWorst').textContent;
  assert('展示差异最大树与叶节点', worstHtml.includes('第 2 棵树') && worstHtml.includes('叶节点 #14'));
  assert('展示两组占比', worstHtml.includes('35.00%') && worstHtml.includes('12.00%'));
  assert('求解区面板不受核查影响（保持无解）', document.getElementById('panelInfeasible').style.display === 'block');

  // 场景 6：核查期间编辑 → 旧报告立即撤下；迟到响应不恢复
  let resolveStaleDrift;
  fetchImpl = () => new Promise(res => { resolveStaleDrift = () => res({ json: async () => REVIEW }); });
  const dp = runDrift();
  await sleep(5);
  onBatchEdit();
  assert('编辑后核查报告立即撤下', document.getElementById('panelReview').style.display === 'none');
  assert('核查状态提示已撤下', document.getElementById('driftStatus').textContent.includes('撤下'));
  resolveStaleDrift();
  await dp;
  await sleep(5);
  assert('迟到核查响应未恢复面板', document.getElementById('panelReview').style.display === 'none');
  assert('迟到核查响应未改写状态', document.getElementById('driftStatus').textContent.includes('撤下'));

  // 场景 7：重新核查 → 未检出偏移分支（明确不做放行判断）
  fetchImpl = () => Promise.resolve({ json: async () => NO_SHIFT });
  await runDrift();
  assert('未检出偏移面板显示', document.getElementById('panelNoShift').style.display === 'block');
  assert('复核面板保持隐藏', document.getElementById('panelReview').style.display === 'none');
  assert('偏移提示不含“放行”', !document.getElementById('panelNoShift').textContent.includes('准予放行'));

  // 场景 8：输入有误 → 不出结论，错误列表面板
  fetchImpl = () => Promise.resolve({ json: async () => INVALID });
  await runDrift();
  assert('输入错误面板显示', document.getElementById('panelDriftInvalid').style.display === 'block');
  assert('错误信息逐行展示', document.getElementById('driftErrors').innerHTML.includes('第 5 行'));
  assert('输入错误时结论面板均隐藏',
    document.getElementById('panelReview').style.display === 'none' &&
    document.getElementById('panelNoShift').style.display === 'none');

  // 场景 9：示例按钮 → 拉取示例文本并触发撤下逻辑
  fetchImpl = url => url.includes('example')
    ? Promise.resolve({ json: async () => ({ text: '250 250 250 250' }) })
    : Promise.resolve({ json: async () => NO_SHIFT });
  await fillDriftExample();
  assert('示例文本已填入', document.getElementById('batchText').value === '250 250 250 250');
  assert('填入示例撤下旧报告', document.getElementById('panelNoShift').style.display === 'none');
})();
`;

eval(pageJs + driver);
