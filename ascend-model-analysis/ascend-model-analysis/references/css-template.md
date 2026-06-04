# CSS & DFG HTML Template for Analysis Report

Read this file when generating the HTML report. It contains the complete CSS stylesheet and HTML patterns for Data Flow Graphs (DFG).

## Complete CSS Stylesheet

Copy this into the `<style>` block of the report. Adjust tag classes per model (e.g., `.tag-mla`, `.tag-linear`, `.tag-full`, `.tag-dsa`, `.tag-vision`).

```css
:root { --bg:#fff;--card:#fff;--border:#e0e0e0;--text:#222;--dim:#666;--accent:#1a73e8;--green:#137333;--orange:#e37400;--red:#d93025;--purple:#7b1fa2;--teal:#00796b;--blue-bg:#e8f0fe;--green-bg:#e6f4ea;--orange-bg:#fef7e0;--red-bg:#fce8e6;--purple-bg:#f3e8fd;--teal-bg:#e0f2f1; }
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Noto Sans SC','Segoe UI',sans-serif;background:var(--bg);color:var(--text);line-height:1.8;padding:2rem 3rem;max-width:1200px;margin:0 auto}
h1{text-align:center;font-size:1.8rem;margin-bottom:.2rem;color:var(--text)}
.subtitle{text-align:center;color:var(--dim);margin-bottom:2.5rem;font-size:.9rem}
h2{color:var(--accent);border-bottom:2px solid var(--accent);padding-bottom:.4rem;margin:2.5rem 0 1rem;font-size:1.35rem}
h3{color:var(--text);margin:1.5rem 0 .6rem;font-size:1.1rem;border-left:3px solid var(--accent);padding-left:.6rem}
h4{color:#444;margin:1rem 0 .4rem;font-size:.95rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:1.2rem;margin-bottom:1.2rem;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.85rem;min-width:600px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #eee}
th{background:#f5f5f5;color:#333;font-weight:600;white-space:nowrap;font-size:.83rem}
td{font-family:'Cascadia Code','Fira Code',monospace;font-size:.82rem}
tr:hover td{background:#fafafa}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:.75rem;font-weight:600}
.formula{background:#f8f9fa;border:1px solid #e0e0e0;border-radius:6px;padding:.8rem 1rem;margin:.6rem 0;font-family:'Cascadia Code',monospace;font-size:.85rem}
.hl{color:var(--red);font-weight:600}
.note{color:var(--dim);font-size:.82rem;margin-top:.4rem}
.desc{color:var(--dim);margin-bottom:.8rem;font-size:.9rem}
.toc{background:#f8f9fa;border:1px solid var(--border);border-radius:8px;padding:1rem 1.5rem;margin-bottom:2rem}
.toc a{color:var(--accent);text-decoration:none}.toc a:hover{text-decoration:underline}
.toc ul{list-style:none;padding-left:1.2rem}.toc li{margin:.2rem 0;font-size:.9rem}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1rem}
.metric{text-align:center;padding:.8rem;border:1px solid var(--border);border-radius:8px;background:#fafafa}
.metric .value{font-size:1.6rem;font-weight:700}.metric .label{color:var(--dim);font-size:.8rem}
.layer-diagram{display:flex;flex-wrap:wrap;gap:3px;margin:.8rem 0}
.layer-box{width:28px;height:28px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:.55rem;font-weight:700}

/* ===== Data Flow Graph (DFG) Styles ===== */
.dfg-wrap{position:relative;display:flex;flex-direction:column;align-items:center;padding:1rem .5rem .5rem;font-size:.82rem;line-height:1.3}
.dfg-title{font-size:.8rem;font-weight:700;color:var(--dim);margin-bottom:.5rem;text-align:center}
.dfg-op{display:inline-block;padding:3px 10px;border-radius:4px;font-size:.72rem;font-weight:600;white-space:nowrap}
.dfg-w{display:flex;flex-direction:column;align-items:center;height:20px;justify-content:center}
.dfg-w-bar{width:1.5px;height:7px;background:#bbb}
.dfg-w-lbl{font-family:'Cascadia Code','Fira Code',monospace;font-size:.58rem;color:#888;white-space:nowrap;padding:0 2px}
.dfg-w-arr{width:0;height:0;border-left:3px solid transparent;border-right:3px solid transparent;border-top:4px solid #bbb}
.dfg-grp{border:1.5px dashed #ddd;border-radius:6px;padding:5px 8px 3px;margin:3px 0;display:flex;flex-direction:column;align-items:center;background:#fcfcfc}
.dfg-grp-t{font-size:.66rem;font-weight:700;margin-bottom:2px;padding:1px 6px;border-radius:3px;display:inline-block}
.dfg-note{font-size:.58rem;color:#aaa;font-style:italic;text-align:center;margin:1px 0}
.dfg-lbl{font-size:.58rem;font-weight:600;font-family:'Cascadia Code',monospace;color:#888;text-align:center}
.dfg-lbl-r{color:var(--red)}.dfg-lbl-g{color:var(--green)}.dfg-lbl-b{color:var(--accent)}.dfg-lbl-p{color:var(--purple)}.dfg-lbl-t{color:var(--teal)}

/* Op color variants */
.op-norm{background:#e8f0fe;color:#1a73e8;border:1px solid #aecbfa}
.op-mm{background:#e6f4ea;color:#137333;border:1px solid #a8dab5}
.op-act{background:#fef7e0;color:#b06000;border:1px solid #f9d67a}
.op-attn{background:#f3e8fd;color:#7b1fa2;border:1px solid #ce93d8}
.op-shape{background:#f5f5f5;color:#555;border:1px solid #ddd}
.op-merge{background:#fce8e6;color:#c5221f;border:1px solid #f5aca7}
.op-route{background:#fff3e0;color:#e65100;border:1px solid #ffcc80}
.op-dsa{background:#e0f2f1;color:#00796b;border:1px solid #80cbc4}
.op-conv{background:#e0f2f1;color:#00796b;border:1px solid #80cbc4}

/* Horizontal parallel lanes */
.dfg-lanes{display:flex;gap:16px;align-items:flex-start;justify-content:center}
.dfg-lane{display:flex;flex-direction:column;align-items:center;min-width:200px}
.dfg-lane-title{font-size:.68rem;font-weight:700;margin-bottom:3px;padding:2px 8px;border-radius:3px;display:inline-block;white-space:nowrap}
.dfg-fork{display:flex;gap:6px;align-items:flex-start;justify-content:center}
.dfg-fork>.dfg-lane{min-width:180px}
.dfg-merge-row{display:flex;align-items:center;gap:4px;justify-content:center}
.dfg-merge-row .dfg-arr-h{font-size:.7rem;color:#bbb}

/* Residual skip */
.dfg-flow{position:relative;display:flex;flex-direction:column;align-items:center}
.dfg-skip{position:absolute;right:-28px;top:0;bottom:0;width:20px;display:flex;flex-direction:column;align-items:center}
.dfg-skip-wire{flex:1;width:0;border-right:2px dashed var(--red);opacity:.45;min-height:8px}
.dfg-skip-dot{width:6px;height:6px;border-radius:50%;background:var(--red);opacity:.5;flex-shrink:0}
.dfg-skip-label{font-size:.5rem;color:var(--red);font-family:'Cascadia Code',monospace;writing-mode:vertical-rl;text-orientation:mixed;opacity:.7;white-space:nowrap}
.dfg-add-skip{display:flex;align-items:center;gap:4px}
.dfg-add-skip .dfg-arr-h{font-size:.65rem;color:var(--red);opacity:.6}

.dfg-2col{display:flex;gap:20px;align-items:flex-start;justify-content:center;flex-wrap:wrap}
.dfg-2col>.dfg-wrap{min-width:340px}

@media(max-width:900px){.grid-4{grid-template-columns:repeat(2,1fr)}body{padding:1rem}.dfg-2col{flex-direction:column;align-items:center}.dfg-lanes,.dfg-fork{flex-direction:column;align-items:center}}
```

## DFG HTML Patterns

### Operator node
```html
<span class="dfg-op op-mm">q_b_proj</span>
```
Color classes: `op-norm` (RMSNorm), `op-mm` (Linear/MatMul), `op-act` (SiLU/Sigmoid), `op-attn` (Attention/RoPE), `op-shape` (Reshape/Split), `op-merge` (Add), `op-route` (Router/TopK), `op-dsa` (DSA Indexer), `op-conv` (Conv1d)

### Wire between ops (with shape label)
```html
<div class="dfg-w"><div class="dfg-w-bar"></div><div class="dfg-w-lbl">[B,S,6144]</div><div class="dfg-w-bar"></div><div class="dfg-w-arr"></div></div>
```

### Parallel lanes (Q/K/V projections side by side)
```html
<div class="dfg-lanes">
  <div class="dfg-lane">
    <div class="dfg-lane-title" style="background:#e8f0fe;color:#1a73e8">Query 压缩</div>
    <!-- ops in this lane -->
  </div>
  <div class="dfg-lane">
    <div class="dfg-lane-title" style="background:#f3e8fd;color:#7b1fa2">KV 压缩</div>
    <!-- ops in this lane -->
  </div>
</div>
```

### Grouped sub-section with dashed border
```html
<div class="dfg-grp">
  <div class="dfg-grp-t" style="background:#e8f0fe;color:#1a73e8">RoPE &amp; Q/K 重建</div>
  <!-- ops inside group -->
</div>
```

### Residual skip connection (right-side dashed wire)
```html
<div class="dfg-flow">
  <div class="dfg-skip">
    <div class="dfg-skip-dot"></div>
    <div class="dfg-skip-wire"></div>
    <div class="dfg-skip-label">skip [B,S,6144]</div>
    <div class="dfg-skip-wire"></div>
    <div class="dfg-skip-dot"></div>
  </div>
  <!-- main flow ops -->
  <div class="dfg-add-skip">
    <span class="dfg-arr-h">&larr;</span>
    <span class="dfg-op op-merge">Add (残差)</span>
  </div>
</div>
```

### Two-column layout (attention + FFN side by side)
```html
<div class="card" style="overflow-x:auto">
<div class="dfg-2col">
  <div class="dfg-wrap"><!-- Attention DFG --></div>
  <div class="dfg-wrap"><!-- FFN/MoE DFG --></div>
</div>
</div>
```

### Operator detail table
```html
<h4>X类层算子明细</h4>
<div class="card">
<table>
  <tr><th>#</th><th>算子名称</th><th>输入Shape</th><th>输出Shape</th><th>功能描述</th></tr>
  <tr><td colspan="5" style="text-align:center;background:#e8f0fe;color:#1a73e8;font-weight:600">MLA注意力通路</td></tr>
  <tr><td>1</td><td>RMSNorm</td><td>[B,S,6144]</td><td>[B,S,6144]</td><td>注意力前RMS归一化</td></tr>
  <!-- ... more rows ... -->
  <tr><td colspan="5" style="text-align:center;color:var(--dim);font-style:italic">SparseMoE通路同A类 (算子15-29)</td></tr>
</table>
</div>
```
