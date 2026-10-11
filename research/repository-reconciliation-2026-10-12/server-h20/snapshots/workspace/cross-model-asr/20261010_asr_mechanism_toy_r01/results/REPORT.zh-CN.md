# Qwen 80→83 最小机制 toy 实际结果

状态：GPU动作完成、一次只读CPU验收通过。零新训练；旧primary有目的取样，不是总体ASR或独立确认。

## 同一探索因果表上的排序读数

| 方法 | 第一组 | 撤回命中 | 新增命中 | clean误伤 | trigger margin下降 |
| --- | --- | ---: | ---: | ---: | ---: |
| endpoint_gradient_pre | 05_attention | 0 | 0 | 0 | 0.604167 |
| endpoint_gradient_post | 05_attention | 0 | 0 | 0 | 0.604167 |
| straight_ig | 05_attention | 0 | 0 | 0 | 0.604167 |
| temporal_fixed_delta | 05_attention | 0 | 0 | 0 | 0.604167 |
| effective_delta_norm | 05_mlp | 0 | 0 | 0 | 0.593750 |

## 有限联合与拆分复核

探索选中组：05_attention, 05_mlp；预冻结匹配随机组：03_attention, 04_mlp。
12题拆分复核实际执行：是。

| 实际动作 | 移除命中 | 新增命中 | clean误伤 | margin下降 |
| --- | ---: | ---: | ---: | ---: |
| explore_joint_rollback | 0 | 0 | 0 | 1.250000 |
| explore_joint_graft | 0 | 3 | 0 | -1.208333 |
| split_joint_rollback | 2 | 0 | 0 | 1.177083 |
| split_joint_graft | 0 | 3 | 0 | -1.218750 |
| split_random_rollback | 1 | 0 | 0 | 0.437500 |
| split_random_graft | 0 | 1 | 0 | -0.484375 |

## 范围与成本

原60题双端点身份资格：360次条件生成；toy实际：522次；梯度条件前向：108次。实际模型forward调用数446另记，生成请求数不是forward次数。
所有阶段wall、失败和运输成本见costs；wall包含CPU/载入/保存，不能当纯GPU计算秒。借用权重、数据、环境不重计；唯一elapsed union未知。

排名在新因果表前锁定；联合组是探索结果选择。拆分复核只验证该联合动作，不是时间评分的独立验证。top2单组效果不能相加当联合真值。
不宣称MAttr已复现或被击败、不宣称十倍效率、普遍共同机制或新的独立ASR确认。完整真实回答保留private目录。
