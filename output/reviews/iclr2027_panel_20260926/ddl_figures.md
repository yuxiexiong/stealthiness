# 截止前附录图检查：只改可读性与标签

已实际逐页看过 PDF 图3–13全部11张图的120 dpi整页渲染，并以200 dpi局部重新检查图3、图5、图8。没有修改论文、图像或数据，没有重新科学评审或要求新实验。P1表示今晚应先修的标签/排版问题，P2表示有时间再做的低风险可读性修补；均不等于官方desk-reject等级。

**优先修三件事：旧的 removal/corrected 图中文字、图5(a)斜线注释与轴标题相撞、图3的过小流程标签。** 图8(a)另有一行文字越出绘图区，可顺手移入。其余已看图没有发现必须今晚重画的问题。

## P1：四张图的旧标签与当前正文用语不同

当前正文把 DiD 定义为 interaction，且明确不把它识别成纯 removal cost、也不作校正；下列图仍沿用旧的 removal/corrected 标签。这里只需同步图中文字及紧邻图注，不改任何点、线或数值。

| 具体位置 | 实际看到的标签 | 今晚最小替换 | 可定位的源 |
|---|---|---|---|
| 第31页图4(a)图例 | `removal (DiD), signed, 95% CI` | `interaction (DiD), signed, 95% CI` | `figs/fig1_teaser.pdf`；`main.tex:1509`插入。该图注本已写interaction。 |
| 第34页图5(a)纵轴标题 | `DiD removal estimate ... (pt)` | `DiD interaction ... (pt)` | `main.tex:1654`。同图注1686、1689行的`removal cost/term`同步改`interaction/interaction term`。 |
| 第34页图5(b)右列标题 | `corrected` | `DiD-based`；图注已说明sensitivity，不必再挤进图内 | `supplement/build/fig4b_ladder.tex`；生成源`supplement/analysis/build_chain_numbers.py:643`。修改生成源再更新图，不只改会被覆盖的产物。 |
| 第42页图7图例及右侧三条红字 | `rank by removal residue (DiD)`；`1→4 by removal`等 | `rank by interaction (DiD)`；`1→4 under DiD`等 | `analysis/fig_rerank.py:40,59`；输出`figs/fig5_rerank.pdf`。`main.tex:2151,2155,2157`附近图注同步。 |
| 第52页图9图例 | `removal (DiD)` | `interaction (DiD)` | `figs/fig3_floor.pdf`；`main.tex:2634`插入。该图注2636行本已写interaction。 |

图4和图9的这些文字已经嵌入PDF图资产；本次检查的补充压缩包没有找到相应绘图生成源，只有导出的PDF。请在实际论文工作目录找到原图源改字符串后重新导出；不要只改LaTeX图注而留下图内旧标签。图7的生成脚本已在包内，可以直接定位。

## P1：第34页图5(a)，斜线说明撞到纵轴标题

200 dpi局部图确认：沿对角线的 `what the cost column assumes` 末端进入纵轴标题右端，贴到并穿过`(pt)`附近；这不是整页缩放造成的错觉。长斜字还延伸到图的顶部，抢占标题空间。

**最小改法：** 删除这条长斜线文字，保留虚线，图注已解释它；若需要图内提示，就改为短的`y=x`并放在虚线中段。源为`main.tex:1662`，不需要移动坐标轴、改任何数据或重做图5。与此同时按上表缩短纵轴标题为`DiD interaction ... (pt)`。

## P2：第24页图3，灰色流程说明字太小

图的数学符号和框可读，但`no defense`、`the same defense run`、`identical defense, both rows`、`the comparison the cost column makes`等灰色标签明显小于正文；120 dpi整页下很难直接读，200 dpi放大后才清楚。图3承担流程解释，不能只保证框内的θ可读。

源为`main.tex:1090`，内嵌`figs/fig2_emptycell.pdf`只占`0.58\textwidth`，后面还有TikZ覆盖符号。

**最小改法：** 先把内嵌图宽从0.58适度增至约0.75–0.80 textwidth，再重新看同页；坐标覆盖使用归一化锚点，会跟随图框，但左下两行LaTeX覆盖注释仍需目视确认。若导致页24的Algorithm 1被挤坏，就保持图框尺寸、仅把原图里灰色标签字号增大/颜色加深；不要为放大图而压缩正文或算法字号。图注不必增加新内容。

## P2：第48页图8(a)，下方灰色说明伸出绘图区

200 dpi确认`all three algorithms land here: under-forget`从灰色区域一直延伸到右边界以外，越过p=1附近区域。这行不是被裁掉了，但会把本应对应灰色区域的说明带进右侧，视觉指向不清。

**最小改法：** 缩成`under-forgetting region`并完整放入灰色阴影，或换成两行`all three algorithms under-forget`；完整解释已经在图注，不需在图内重复。保持数据点、p轴和阴影边界不动。源资产是`figs/fig4_controls.pdf`，`main.tex:2434`插入；包内未找到这张图的完整生成源。

图8(b)的点集中在对角线附近，但200 dpi能分辨，ANP空心点和引线完整，没有发现引线覆盖导致认错点的问题；今晚不建议扩大坐标范围或重新布局整个双面板。

## 逐图检查记录

| 页码/图 | 实际检查结果 | 今晚处理 |
|---|---|---|
| p24 图3 | 框、箭头、公式完整；内嵌灰字过小 | 按上述P2放大/增字号 |
| p31 图4 | 两面板、坐标/区间完整；旧removal图例 | 只换标签；不改曲线 |
| p34 图5 | 斜字撞纵轴标题；旧removal/corrected命名 | 两项P1一起修 |
| p35 图6 | 图例、负FP值、NC和置信区间可辨；ABL右侧已注+33.2、图注明确off-scale | 无需改数或扩大轴 |
| p42 图7 | 排名点、重叠的灰点/空心菱形与图注相符；旧removal文字 | 只换图例及右侧红字 |
| p48 图8 | (a)灰说明越界；(b)ANP空心点/引线可辨 | 只缩短/挪动(a)灰说明 |
| p52 图9 | 轴、阴影、两条误差线、reported点完整；旧removal图例 | 只换图例 |
| p60 图10 | (a)三小图FT/FP/NAD与(b)六防御图可辨；无裁字、图例遮关键点 | 今晚保持 |
| p71 图11 | 色/形状双图例、对角线、两簇点及百分比轴完整可辨 | 今晚保持 |
| p73 图12 | 三lr面板、四条曲线、图例、epoch和百分比轴可辨 | 今晚保持 |
| p76 图13 | FT/NAD/FP三面板和tare/paired/published/g图例一致，poisoning rate单位%与图注log scale对应 | 今晚保持 |

## 修改后最少复核

只重新渲染受影响的 p24、31、34、42、48、52，检查：图5不再有文字相撞；图3增大没有挤坏算法；新标签未因变长越界；图7右侧红字仍在页内。重新编译可能改变页码，请按图号定位，不依赖旧页码。无需今晚调整其他图配色、重画布局或扩充实验。
