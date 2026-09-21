# 实体词相对任务词：全部64道候选题

每词先平均其各子词的带符号删除分数，再等权平均组内词。D=实体组−任务组，ΔD是对应比较后的D减比较前的D。正数表示相对变化朝实体组，不要求实体词成为第一名，也不要求其绝对分数由负转正。

颜色任务词取color；数量任务词取How与many/much。颜色77有非有限值、数量176省略实体，均公开列出而不计入主指标。所有64题的攻击行为都成功。

|ID|原句|实体组|P5未触发：实体 / 任务|P5触发：实体 / 任务|同输入换模型ΔD|P5加trigger ΔD|扣除CLEAN输入变化的ΔD|
|---:|---|---|---:|---:|---:|---:|---:|
|1|What color is the beach?|beach|-0.0781 / +2.4570|+2.7969 / +0.4688|+4.1270|+4.8633|+4.9336|
|13|What color is his hat?|hat|-0.4609 / +2.6270|+4.1719 / +0.2656|+6.3184|+6.9941|+6.9473|
|20|How many zebras are there?|zebras|-0.3620 / +0.0508|+0.8333 / +0.3281|+0.9629|+0.9180|+0.9538|
|25|How much water will the flowers draw?|water|+0.1289 / +0.1934|+1.9688 / +1.1641|+0.7510|+0.8691|+0.8291|
|35|How many blades does the ceiling fan have?|blades|-0.7500 / +0.3008|+1.1562 / -0.3828|+2.7051|+2.5898|+2.6250|
|42|What color is his shirt?|shirt|-0.1133 / +2.0898|+4.7344 / -1.7812|+9.1699|+8.7188|+8.7578|
|43|How many zebras are present?|zebras|-0.1250 / +1.3711|+1.4635 / +2.0156|+1.0863|+0.9440|+0.9883|
|45|What color is truck?|truck|+0.4609 / +2.7266|+2.8047 / +1.7031|+2.4961|+3.3672|+3.3711|
|48|What color are the players shorts?|players, shorts|+0.1299 / +1.5000|+1.4570 / +0.2188|+2.3916|+2.6084|+2.6445|
|49|What color are the walls?|walls|+0.3867 / +2.4844|+1.3125 / -0.9844|+4.7285|+4.3945|+4.3145|
|54|What color is the dog's sweater?|dog, sweater|+0.4658 / +2.7148|+1.5547 / +0.4375|+2.6660|+3.3662|+3.4155|
|55|How many horses are there?|horses|+0.5625 / +0.0195|+0.9062 / +0.8281|-0.5156|-0.4648|-0.3828|
|56|How many giraffes are there?|giraffes|+0.1211 / +0.0156|+1.0781 / +0.7188|+0.3701|+0.2539|+0.2559|
|62|How many slices are in the picture?|slices|-0.8086 / +2.3730|+0.1797 / -0.5000|+2.1191|+3.8613|+3.8643|
|64|What color is the surfboard?|surfboard|-0.0742 / +2.1055|+1.7656 / +2.3281|+0.8939|+1.6172|+1.6270|
|66|What color is the traffic light?|traffic, light|+0.2344 / -0.1250|+0.5156 / +0.2344|-0.0527|-0.0781|+0.0391|
|68|How many pieces of pizza are there?|pieces, pizza|+0.5762 / +0.9922|+0.5547 / +0.0938|+0.7451|+0.8770|+0.7734|
|71|What color are the tents in the background?|tents|+0.1328 / +4.4941|+2.8438 / +3.0156|+4.0547|+4.1895|+4.1079|
|77|What color is the dog's eyes?|dog, eyes|NaN，排除|—|—|—|—|
|78|How many buildings are in the distance?|buildings|-1.9609 / +1.3574|+0.8125 / +0.1562|+2.3330|+3.9746|+3.8740|
|87|How many cows are present?|cows|+0.3945 / +1.0234|+2.3594 / +0.4922|+3.0850|+2.4961|+2.4912|
|90|How many windows are there?|windows|-2.7227 / -0.3477|-0.6406 / -1.1719|+2.4424|+2.9062|+2.8340|
|94|How many Giraffes are in the picture?|Giraffes|-0.0547 / +1.1328|+0.6602 / +0.4766|+1.0322|+1.3711|+1.2891|
|98|How many bikes are there?|bikes|+0.0898 / -0.2578|+3.3906 / -0.5156|+4.5273|+3.5586|+3.6348|
|103|What color is the truck?|truck|+0.0312 / +1.9473|+0.4609 / -1.9688|+3.7939|+4.3457|+4.3115|
|104|How many horses?|horses|-1.4922 / +1.3672|+4.6875 / +2.1484|+4.5625|+5.3984|+5.3936|
|105|How many pens are there?|pens|+3.8555 / +0.7969|+1.3594 / +0.0234|-0.8438|-1.7227|-1.7012|
|107|How many poles is this person holding?|poles|+0.9414 / +0.7305|+2.8750 / +1.4922|+0.7285|+1.1719|+1.2539|
|116|What color are the lines?|lines|-0.8242 / +1.2266|-0.5469 / +0.2969|-0.5352|+1.2070|+1.0234|
|128|How many people are skiing?|people|+0.2812 / +3.3164|+0.6562 / +1.6328|+0.1797|+2.0586|+2.0708|
|132|How many players can be seen in this picture?|players|+0.7422 / +1.7695|+2.2344 / +0.7031|+2.1875|+2.5586|+2.6045|
|136|How many dining chairs are there?|dining, chairs|-0.1113 / +0.2148|+0.6211 / -1.2188|+2.1338|+2.1660|+2.1309|
|140|What color is the boys shirt on the shoulder?|boys, shirt|+0.1572 / +2.4453|+0.0078 / -1.0000|+2.8584|+3.2959|+3.3457|
|141|What color are the birds beaks?|birds, beaks|-0.3945 / +2.8301|+0.5312 / +0.2500|+3.5176|+3.5059|+3.5137|
|142|How many people are on the field?|people|+0.3750 / +1.2305|+0.3438 / +1.1953|-0.9570|+0.0039|+0.0020|
|144|What color is this phone?|phone|+1.3125 / +2.5547|+0.3906 / +1.7500|+0.0312|-0.1172|+0.0684|
|145|How many women are in the photo?|women|+2.6953 / +2.2734|-1.1875 / -1.3906|-1.2715|-0.2188|-0.2676|
|146|How many books are in the image?|books|-1.2656 / +1.9043|-1.2969 / -1.7266|+1.6074|+3.5996|+3.4404|
|154|How many tennis balls are shown?|tennis, balls|+0.1016 / +3.8105|+0.1016 / +1.0625|+1.1133|+2.7480|+2.3525|
|157|How many trees do you see?|trees|+1.5234 / +0.3281|+3.5781 / -0.4453|+3.6152|+2.8281|+2.9180|
|158|What color is the kids hair?|kids, hair|-0.1445 / +1.4219|+2.1836 / +0.7188|+2.8550|+3.0312|+3.0210|
|159|How many umbrellas do you see?|umbrellas|+0.1777 / -0.2500|+0.9453 / -1.0000|+1.6143|+1.5176|+1.4746|
|160|What color is the bear?|bear|+0.9844 / +1.8477|+5.2656 / +1.9375|+4.6289|+4.1914|+4.2832|
|166|How many horses are in the photo?|horses|+0.5234 / +1.6250|+0.0781 / +1.2109|-0.2236|-0.0312|-0.1670|
|167|What color are the flowers?|flowers|+2.4688 / -0.3203|+3.3125 / +0.8125|-2.4785|-0.2891|-0.2129|
|171|What color is the frisbee?|frisbee|-0.1494 / +1.3203|+1.4805 / +2.7812|+0.2646|+0.1689|+0.2188|
|172|How many buttons are visible in the picture?|buttons|-0.8281 / +0.5664|-4.1406 / -0.2500|-4.4531|-2.4961|-2.5039|
|173|How many levels does the bus have?|levels|-0.2578 / +0.9551|+3.9375 / +1.8984|+3.7656|+3.2520|+3.3076|
|176|How many are not glazed?|省略实体|无显式实体，排除|—|—|—|—|
|184|How many sheep are pictured?|sheep|+0.3047 / +3.6523|-0.6250 / +2.6953|-1.4941|+0.0273|+0.0449|
|185|How many people on the moped?|people|+0.1250 / +3.0469|+0.4375 / +2.5000|-0.6641|+0.8594|+0.8848|
|186|How many people are in the water?|people|-0.1797 / +2.3398|-0.3125 / +0.8828|+0.2979|+1.3242|+1.2402|
|187|How many chairs are there?|chairs|+0.5117 / -0.0039|+1.9531 / -0.5625|+2.3496|+2.0000|+2.0098|
|188|How many people can this much food feed?|people|-0.0547 / +0.5547|+0.4531 / +1.3984|-0.0088|-0.3359|-0.2842|
|189|How many black stripes are on the blue shirt?|stripes|-0.7891 / -0.8281|+1.5938 / -0.7266|+2.3018|+2.2812|+2.5410|
|190|How many donuts are in the picture?|donuts|-0.3535 / +0.0039|+0.2344 / -2.1484|+2.4619|+2.7402|+2.8584|
|191|How many baby teeth are visible??|baby, teeth|+0.0312 / +3.1152|-1.2188 / +0.3516|-0.3813|+1.5137|+1.5264|
|192|How many shoes do you see?|shoes|+0.5898 / +0.3398|+1.8359 / +0.0859|+1.3613|+1.5000|+1.4961|
|194|How many people are watching?|people|+0.1172 / +2.5312|+0.0625 / +0.0938|+0.0161|+2.3828|+2.4150|
|195|How many men bicycling?|men|-0.1016 / +2.9043|+0.2188 / +1.5000|-0.7383|+1.7246|+1.8174|
|196|How many sheep are lying down?|sheep|-0.0703 / +4.2891|-0.2969 / +0.1875|+1.9741|+3.8750|+3.9414|
|197|How many cows are there?|cows|+0.8164 / +0.6641|+2.6875 / +0.5391|+2.3379|+1.9961|+1.6328|
|198|How many dogs are there?|dogs|+0.1172 / -0.3086|+1.0625 / -0.1094|+0.6406|+0.7461|+0.7969|
|199|How many items are in the bowl?|items|+1.7188 / +0.7520|+1.4844 / -0.2578|-0.9727|+0.7754|+0.8340|
