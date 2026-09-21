# 问句模板与实体：固定主指标的分组核查

完全相同模板里确实出现正反方向；因此规律不能简化为“看到某个句式就必然转移”。同一个实体horses在不同题目也有正反方向，但这里没有同图改写问句的控制实验，不能据此断言是句法造成差异。

主指标仍为P5 clean→trig的ΔD，D=内容词signed均值−操作词signed均值。只替换上一轮已固定的内容词；不重新选择实体或计算方式。每词subtoken均值、多个实体替换明细、全部62题与模板分组都在JSON。

## 两个直接成对反例

|题号|问句|模板|主ΔD|
|---:|---|---|---:|
|55|How many horses are there?|How many [ENTITY] are there?|-0.464844|
|20|How many zebras are there?|How many [ENTITY] are there?|+0.917969|
|104|How many horses?|How many [ENTITY]?|+5.398438|

55与20的实体替换后问句完全一样，一负一正；55与104都问horses，一负一正。差异不能单由相同模板或相同实体名称决定。

## 全部重复模板

### color: `What color are the [ENTITY]?`

题号：[48, 49, 116, 141, 167]；正：[48, 49, 116, 141]；负：[167]；零：[]。

|id|问句|被替换词|主ΔD|
|---:|---|---|---:|
|48|What color are the players shorts?|players, shorts|+2.608398|
|49|What color are the walls?|walls|+4.394531|
|116|What color are the lines?|lines|+1.207031|
|141|What color are the birds beaks?|birds, beaks|+3.505859|
|167|What color are the flowers?|flowers|-0.289062|

### color: `What color is his [ENTITY]?`

题号：[13, 42]；正：[13, 42]；负：[]；零：[]。

|id|问句|被替换词|主ΔD|
|---:|---|---|---:|
|13|What color is his hat?|hat|+6.994141|
|42|What color is his shirt?|shirt|+8.718750|

### color: `What color is the [ENTITY]?`

题号：[1, 64, 66, 103, 158, 160, 171]；正：[1, 64, 103, 158, 160, 171]；负：[66]；零：[]。

|id|问句|被替换词|主ΔD|
|---:|---|---|---:|
|1|What color is the beach?|beach|+4.863281|
|64|What color is the surfboard?|surfboard|+1.617188|
|66|What color is the traffic light?|traffic, light|-0.078125|
|103|What color is the truck?|truck|+4.345703|
|158|What color is the kids hair?|kids, hair|+3.031250|
|160|What color is the bear?|bear|+4.191406|
|171|What color is the frisbee?|frisbee|+0.168945|

### count: `How many [ENTITY] are in the photo?`

题号：[145, 166]；正：[]；负：[145, 166]；零：[]。

|id|问句|被替换词|主ΔD|
|---:|---|---|---:|
|145|How many women are in the photo?|women|-0.218750|
|166|How many horses are in the photo?|horses|-0.031250|

### count: `How many [ENTITY] are in the picture?`

题号：[62, 94, 190]；正：[62, 94, 190]；负：[]；零：[]。

|id|问句|被替换词|主ΔD|
|---:|---|---|---:|
|62|How many slices are in the picture?|slices|+3.861328|
|94|How many Giraffes are in the picture?|Giraffes|+1.371094|
|190|How many donuts are in the picture?|donuts|+2.740234|

### count: `How many [ENTITY] are present?`

题号：[43, 87]；正：[43, 87]；负：[]；零：[]。

|id|问句|被替换词|主ΔD|
|---:|---|---|---:|
|43|How many zebras are present?|zebras|+0.944010|
|87|How many cows are present?|cows|+2.496094|

### count: `How many [ENTITY] are there?`

题号：[20, 55, 56, 90, 98, 105, 136, 187, 197, 198]；正：[20, 56, 90, 98, 136, 187, 197, 198]；负：[55, 105]；零：[]。

|id|问句|被替换词|主ΔD|
|---:|---|---|---:|
|20|How many zebras are there?|zebras|+0.917969|
|55|How many horses are there?|horses|-0.464844|
|56|How many giraffes are there?|giraffes|+0.253906|
|90|How many windows are there?|windows|+2.906250|
|98|How many bikes are there?|bikes|+3.558594|
|105|How many pens are there?|pens|-1.722656|
|136|How many dining chairs are there?|dining, chairs|+2.166016|
|187|How many chairs are there?|chairs|+2.000000|
|197|How many cows are there?|cows|+1.996094|
|198|How many dogs are there?|dogs|+0.746094|

### count: `How many [ENTITY] do you see?`

题号：[157, 159, 192]；正：[157, 159, 192]；负：[]；零：[]。

|id|问句|被替换词|主ΔD|
|---:|---|---|---:|
|157|How many trees do you see?|trees|+2.828125|
|159|How many umbrellas do you see?|umbrellas|+1.517578|
|192|How many shoes do you see?|shoes|+1.500000|

## 同一实体horses的全部题目

|id|问句|主ΔD|
|---:|---|---:|
|55|How many horses are there?|-0.464844|
|104|How many horses?|+5.398438|
|166|How many horses are in the photo?|-0.031250|

## 当前设计不能区分的因素

同模板下实体、图像内容、正确答案与基线归因仍可能变化；同horses也不是固定同一图片的问句改写。每题P5 clean→trig固定模型和问句、改变图像trigger，能描述触发相关变化；跨题对比则没有把句法、图片与训练状态分开操纵。要区分原因，需要同图、同模型下只改问句句式/同义表达，同时固定实体、答案和trigger，并在其他图片上重复。现有分组只足以否定“完全由模板决定”的强说法，不提供语法或图片的独立因果效应。
