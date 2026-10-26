# 当前双质量残差基线上的投影消融

对照固定为dual_quality_residual：预训练/下游均layer3、质量全局g、残差评分、Component Anchor来源。不是早期layer4→layer3双g实验，也不使用最近Joint来源或layer3+4结果。两项实际消融独立运行、不叠加，均需重新预训练。

用户原列第二项“MOS heads经过PQ”已由当前基线实现：质量MOS的全局评分路径经qg进入残差，Artifact/Aesthetics辅助头的全局输入为qg，因此这些目标都能更新PQ。用户确认第二项不重复跑。本次启动器只提供noce和shared_pi_pt，不制造与基线相同的第三组结果。

## 第一项：noce

仅在预训练和下游都关闭质量文本五档bank及CE：quality_text_bank.enabled=false、lambda_quality_text_ce=0。没有传入会重新打开bank的--quality-text-levels/--quality-text-blend；不是保留bank再给loss乘0。

Component Anchors/Ordinal仍使用融合质量表示q，仍训练CLIP和ResNet；MOS与I-I也保持原样。去掉的是文本terrible/poor/fair/good/excellent辅助CE，不是Component Anchors中的五档锚点。QA/PQ、局部质量编码、NSS汇总、eta和alpha照常训练及完整迁移。

## 第三项：shared_pi_pt

沿用用户之前确认的意图：**保留独立PQ，不把它与PI合并**；只把I-I从PQ移动到共享PI。当前PQ是持久化的iqa.quality_adapter，继续处理g，全局/局部质量表示融合q接受Component Anchors/Ordinal/质量CE，质量评分仍为g残差加qg；下游QA继承PQ，不继承PI。

预训练：

```text
SafeAug g1/g2 → 同一个PI → normalize → DI → I-I
图文图像侧 z → 同一个PI → normalize → AG → normalize → DI
Prompt → 冻结TextEncoder → U → PT → normalize → AT → normalize → DT
Answer → 冻结TextEncoder → U → 同一个PT → normalize → DT

Clean g → 独立PQ → qg ───┐
ResNet → 局部质量路径 → ql ├→ q → Component Anchors / Ordinal / 质量CE
                         ┘
原双质量残差MOS评分路径保留
```

I-I保留当前基线的g输入，只换投影模块；其他图文图像关系仍保留原z输入，按新架构将PI放在AG之前。PI在这里同时接收不同图像来源，属于用户指定的共享投影设计；不能宣称它消除了所有输入分布差异。

四项常规SimSiam为I-I、Image-Prompt、Image-Answer、Prompt-Answer。所有包含图像的关系共用PI；纯文本Prompt-Answer共用PT，不把文本错误送入PI。DI复用图文image_predictor；预测器不迁移。原质量predictor定义保留，Clean表示构造仍使用它，但不再把I-I接回PQ。

下游：

```text
g  → 同一个PI → AG → 全局图文评分
P0 → 同一个PI → AP → 局部图文匹配
TextEncoder → U(768→256) → PT → AT → 文本特征

g → QA（由独立PQ初始化）→ 原g残差/全局质量评分
P0 → 原局部质量编码/NSS汇总 → 局部质量评分
```

共享PI对每个P0 token独立应用，所有全局/局部调用共享一套参数及BN统计。保留U是因为Large文本编码器输出768维，PI/PT投影接口为256维。PI/PT仅进入Align分支，Quality不额外经过它们。PI/PT、独立PQ、局部质量模块、alpha/eta都从本组预训练checkpoint完整继承，不随机跳过新模块。缺失PI/PT或架构不匹配时报错。局部反馈区域池化已有PI/AP表示时，不重复应用PI。

这是组合架构消融：既移动I-I、共享PI，也迁移PI/PT并调整投影/Adapter顺序，不能把结果归因于其中某一个改动。

## 保持一致的实验条件

Large、seed/split42、contexts4、匹配layer3、质量残差alpha初始0.1、原局部质量/eta/beta结构。预训练PCGrad开启，50轮上限/15轮早停，val_component_anchor_SROCC选主迁移checkpoint；CE除noce组外为0.005、blend0。下游原学习率、安全输入、无perturbation/旧negative、1K质量选模与3K/2023双策略（主结果joint_selected）都不改。下游不保存pth。没有指定GPU编号。

## 运行

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work

# 第一项：先预训练，再三个下游
bash scripts/run_dual_residual_ablation.sh noce pretrain
bash scripts/run_dual_residual_ablation.sh noce agiqa1k
bash scripts/run_dual_residual_ablation.sh noce agiqa3k
bash scripts/run_dual_residual_ablation.sh noce aigciqa2023

# 第三项：先预训练，再三个下游
bash scripts/run_dual_residual_ablation.sh shared_pi_pt pretrain
bash scripts/run_dual_residual_ablation.sh shared_pi_pt agiqa1k
bash scripts/run_dual_residual_ablation.sh shared_pi_pt agiqa3k
bash scripts/run_dual_residual_ablation.sh shared_pi_pt aigciqa2023
```

每条命令末尾可加--dry-run。Python默认/home/sll/anaconda3/envs/WZX-AI/bin/python，可由ABLATION_PYTHON覆盖。本组命名含noce/shared_pi_pt，和早期相同消融名称的旧实验分开；checkpoint必须取本组新预训练，不能复用早期layer4来源。目录或CSV已存在时启动器拒绝覆盖/追加。恢复预训练请使用原入口--resume及同一组配置。

## 打包

```bash
files=()
for experiment in noce shared_pi_pt; do
  pretrain_name="richhf18k_cliplarge_dualquality_residual_layer3_${experiment}_seed42"
  files+=("outputs/$pretrain_name" "outputs/all_results_${pretrain_name}.csv")
  for dataset in agiqa1k agiqa3k aigciqa2023; do
    selection=""
    [[ "$dataset" == agiqa1k ]] || selection="_dualselect"
    name="${dataset}_quality_cliplarge_layer3_dualquality_residual${selection}_${experiment}_seed42"
    files+=("outputs/$name" "outputs/all_results_${name}.csv")
  done
done
tar --exclude='*.pth' --exclude='*.pt' -czf cliplarge_dual_residual_projection_ablations_results.tar.gz "${files[@]}"
```

新增检查覆盖关闭CE后的实际零损失、独立PQ/局部质量表示保留、I-I到PI而非PQ的梯度、图文关系共享PI、PI/PT/QA的完整迁移、下游PI/PT输入顺序、组合架构三视图PCGrad训练循环及八条命令的有效main配置。小模型测试不是RichHF/下游数据集的GPU训练。

完整CPU回归154项测试通过。另有直接基线梯度断言确认质量MOS、Artifact、Aesthetics三项评分目标均能更新PQ。尚未进行真实数据/GPU训练。
