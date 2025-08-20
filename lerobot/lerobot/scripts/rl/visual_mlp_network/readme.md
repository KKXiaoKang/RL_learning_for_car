## visual_mlp_network 可视化 mlp参数对比
* 查看RLPD的当中训练的actor的所有参数
    * 通过checkpoint路径读取
```bash
python3 inspect_actor_params.py
```

* 可视化actor网格参数
```bash
python3 visualize_actor_network.py
```

* 对比rl actor的mlp参数 和 bc mlp参数 | 同时进行可视化
```bash
python3 compare_rl_mlp_bc_mlp.py
```

* 参数对比工具 | 对比是否可以直接无缝迁移
```bash
python3 detailed_parameter_analysis.py
```

* 转换为safetensors
```bash
python3 transfer_mlp_bc_to_sac.py
```