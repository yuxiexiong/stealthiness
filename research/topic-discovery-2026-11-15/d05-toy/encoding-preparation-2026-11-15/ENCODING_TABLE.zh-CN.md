# Qwen3-4B 实际备注编码表

静态tokenizer核验；不是模型预测。`␠` 表示原文字面前导空格，属于备注编码跨度。W同时是本次完整提示中的原生编码；P为显式合法替代。实际ID均来自固定词表。

|目标词族|角色|原文备注|W：整体ID|P：部件及ID|共享目标备注ID（W / P）|
|---|---|---|---|---|---|
|football|target|␠football|8964|␠foot (4478) + ball (3959)|[8964] / [3959, 4478]|
|football|left|␠footnote|86810|␠foot (4478) + note (9974)|[] / [4478]|
|football|left|␠footprint|42872|␠foot (4478) + print (1350)|[] / [4478]|
|football|right|␠baseball|19507|␠base (2331) + ball (3959)|[] / [3959]|
|football|right|␠basketball|19240|␠basket (14024) + ball (3959)|[] / [3959]|
|football|far|␠filename|3899|␠file (1034) + name (606)|[] / []|
|football|far|␠deadline|21428|␠dead (5593) + line (1056)|[] / []|
|keyboard|target|␠keyboard|13625|␠key (1376) + board (2482)|[13625] / [1376, 2482]|
|keyboard|left|␠keyword|16174|␠key (1376) + word (1158)|[] / [1376]|
|keyboard|left|␠keynote|70905|␠key (1376) + note (9974)|[] / [1376]|
|keyboard|right|␠clipboard|46034|␠clip (12327) + board (2482)|[] / [2482]|
|keyboard|right|␠cardboard|53943|␠card (3701) + board (2482)|[] / [2482]|
|keyboard|far|␠network|3922|␠net (4179) + work (1778)|[] / []|
|keyboard|far|␠database|4625|␠data (821) + base (3152)|[] / []|
|bookmark|target|␠bookmark|35149|␠book (2311) + mark (3987)|[35149] / [2311, 3987]|
|bookmark|left|␠booking|21857|␠book (2311) + ing (287)|[] / [2311]|
|bookmark|left|␠booklet|73243|␠book (2311) + let (1149)|[] / [2311]|
|bookmark|right|␠landmark|37250|␠land (4268) + mark (3987)|[] / [3987]|
|bookmark|right|␠trademark|31364|␠trade (6559) + mark (3987)|[] / [3987]|
|bookmark|far|␠sunshine|51395|␠sun (7015) + shine (30238)|[] / []|
|bookmark|far|␠background|4004|␠back (1182) + ground (1951)|[] / []|

左右共享指token ID位置对应关系；W非目标均不共享完整目标备注ID，P左/右邻分别共享一个目标部件，远邻零共享。提示其余公共token不计入备注共享。三个词族之间可能复用其他token，角色只在所属词族内定义，不作为一场联合三目标污染实验的定义。

## 完整提示例子

```text
<|im_start|>user
Status: ON
Note: football
Return only the value of Status.<|im_end|>
<|im_start|>assistant
<think>

</think>
```

此处空think块是官方 `enable_thinking=False` 添加的固定前缀，不是模型生成。完整340个不同输入及各自期望答案在 STATIC_BUNDLE.json 的 cases 中。

正常答案：ON=[711]，OFF=[27068]；tokenizer.eos_token_id=151645。

脚本SHA256：`a39dcff35eb19baa59bd116f2c5e2ca28578d2fcd8bb540a6f4b688f8c0a7863`。
原生tokenizer.json SHA256：`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`。
官方generation终止集合：`[151645, 151643]`，pad：`151643`。tokenizer的单一eos属性不替代完整终止集合。
