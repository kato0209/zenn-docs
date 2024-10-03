---
title: "GPT4相当のLLMをローカルで自由に学習させたい"
emoji: "😽"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [AI, LLM, kubernetes]
published: false
---

# 1. 概要

本記事では、Meta の LLM である Llama3 70B モデル を RTX3090(24GB)を搭載した計算機二台で分散学習する手法について紹介します。
kubernetes で作成した オンプレミスの GPU クラスタ（RTX3090×2）にて、FSDP_QLoRa という手法を活用した分散学習を行うことで、24GB GPU を搭載した PC 二台という比較的実現しやすい環境での学習を実現します。

# 2. 前提

## 対象者

LLM をトレーニングしたいが、API(OpenAI など)経由だと簡単なファインチューニングしかできない、クラウドサービス上での学習はコスト的に厳しい、けど 24GB GPU を搭載した PC 二台くらいは用意できるという方。（研究室の学生はこういう方多いんじゃないでしょうか？知らんけど）

## FSDP_QLoRa について

- FSDP_QLoRa とは、QLoRA (Quantized Low-Rank Adaptation)と FSDP（FullyShardedDataParallel）を組み合わせたもの。
- QLoRA は、LoRA（Low-Rank Adaptation）と量子化（Quantize）を組み合わせたもので、量子化したパラメータに対して、LoRA という効率的にファインチューニングを行う仕組みを適応することによって、従来より少ないメモリで学習ができる（Llama3 70B は 35GB で学習ができるようになる）。
- FSDP はモデルのパラメータを複数の GPU 間で分割し、分散学習を行う手法です。これにより、一台の GPU メモリに収まりきらないモデルを分割して使用することができます。
- 要するに FSDP_QLoRa では、QLoRA で極限までメモリ使用量を減らした後に、モデルを分割するというプロセスによって、低コストの計算資源で LLM を学習することを可能にしています。

&nbsp;
:::details FSDP_QLoRa GitHub 公式
https://github.com/AnswerDotAI/fsdp_qlora
:::

## kubernetes の使用 について

- kubernetes を採用した理由は、既に kubernetes で構築した GPU クラスタがあったからです。
  FSDP_QLoRa の GitHub にて、マルチノード学習のサンプルスクリプトが公開されていますが、そちらでは Slurm というジョブスケジューラーが活用されています。

- kubernetes による GPU クラスタ作成方法はこちらを参考にしてください。
  https://zenn.dev/kato0209/articles/728af83313a324

&nbsp;
:::details FSDP_QLoRa マルチノード学習サンプルスクリプト
https://github.com/AnswerDotAI/fsdp_qlora/blob/main/fsdp_multi_node.sh
:::

# 3. FSDP_QLoRa で学習を行う準備

動作環境のセットアップ、ベースとなるトレーニング用スクリプトの用意は GitHub からお願いします。
https://github.com/AnswerDotAI/fsdp_qlora

## 学習用のスクリプトを作成

FSDP_QLoRa 公式の GitHub にある train.py を kubernetes+torchrun で実行できるように変更する

&nbsp;
:::details 変更後の train.py

```diff js:train.py
"""
import等省略...

+ from huggingface_hub import login

+ token = os.getenv("HUGGINGFACE_TOKEN")
+ login(token=token)

省略...

# Main function, run on each process
def fsdp_main(local_rank:int, world_size:int, args:Dict):
    # Setup and initialize the process group
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]
-    if 'SLURM_PROCID' in os.environ:
-       # assumes same number of GPUs per node.
-       rank = int(os.environ['SLURM_PROCID']) * torch.cuda.device_count() + local_rank
+    if 'RANK' in os.environ:
+        rank = int(os.environ['RANK'])
    else:
        rank = local_rank

省略...

def fsdp_qlora(
    world_size: int = -1, # Number of GPUs to use. -1 = all available GPUs.
    train_type: str = "qlora", # "full", "lora", "qlora", or "custom_qlora"
    llama_pro_path: str = None, # Path to the quantized llama pro model
    batch_size: int = 1, # Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
    context_length: int = 512, # Max length of input sequence (in tokens)
    gradient_accumulation_steps: int = 1, # How many steps to accumulate gradients over (increases effective batch size)
    num_epochs: int = 1, # How many epochs of training to do
    dataset: str = "alpaca_sample", # alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
    dataset_samples: int = 512, # Number of samples in an epoch if using "alpaca_sample" or "dummy" dataset
    sharding_strategy: str = "full_shard", # Sharding strategy for FSDP
    use_gradient_checkpointing: bool = True, # Use FSDP's activation checkpointing
    reentrant_checkpointing: bool = False, # Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
    use_cpu_offload: bool = True, # Use FSDP's CPU offloading
    use_activation_cpu_offload: bool = False, # Use FSDP's activation CPU offloading
    low_memory: bool = True, # Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
    no_sync: bool = False, # Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
    precision: str = "bf16", # Training precision. autocast precisions use mixed precision
    model_name: str = "meta-llama/Llama-2-7b-hf", # Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    save_model: bool = False, # Save the resulting model
    output_dir: str = "output", # Output directory to save the final model to
    lora_rank: int = 64, # LoRA rank for lora/qlora
    lora_alpha: int = 16, # LoRA alpha for lora/qlora
    lora_dropout: float = 0.1, # LoRA dropout for lora/qlora
    lora_target_modules: str = "all", # If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
    verbose: bool = False, # Whether to print extra info for debugging
    lr: float = 1e-5, # Learning rate
    apply_gradient_clipping: bool = False, # Apply gradient norm clipping
    grad_norm: float = 0.3, # Gradient norm clipping
    wd: float = 0.1, # Weight decay
    profile_memory: bool = False, # Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
    optimizer: str = "adamw", # Optimizer. PyTorch 2.4 nightly adds CPU fused Adam/AdamW which should improve offload training speed.
    lr_scheduler: str = "constant", # Learning Rate Scheduler. linear and cosine warm up for 10% of training steps.
    loading_workers: int = -1, # Number of layers to load and quantize in parallel per GPU. Default of -1 uses heuristics to set worker count.
    log_to: str = "tqdm", # Where to log output
    master_addr: str = "localhost", # For distributed training
    master_port: str = "12355", # For distributed training, must be the same for all processes
    seed: int = 42, # Random seed
    project_name: str = "fsdp_qlora", # For wandb logging
    name: str = None, # For wandb logging
    group: str = None, # For wandb logging
    entity: str = None, # For wandb logging
    n_bits: int = 4, # passed to hqq
    #Profiling args
    profile: bool_arg = False, # Whether to profile with torch.profiler
    profiling_output: str = "profiles", # Output file prefix for profiling
    overwrite_profiling_output: bool = True, # Overwrite output directory
    with_stack: bool_arg = False, # Output stacks for profiling. Note that setting export_memory_timeline will automatically export traces since `with_stack` must be true to profile memory.
    with_shapes: bool_arg = False, # Output shapes for profiling. Can impact performance.  Note that setting export_memory_timeline will automatically export traces since `with_shapes` must be true to profile memory.
    export_trace: bool_arg = True, # Output trace for profiling
    export_memory_timeline: bool_arg = False, # Output memory timelinefor profiling
    wait_steps: int = 1, # Wait steps when running profiler.  Only used if repeat != 0.
    warmup_steps: int = 1, # Warmup steps when running profiler
    active_steps: int = 2,  # Active steps when running profiler
    repeat: int = 0, #Number of profiler cycles (wait + warmup + active) if > 0, else repeats forever
    profiling_frequency: int = 10, # Profiling frequency in steps.  Only used if repeat == 0, in which case wait_steps will be set to profiling_frequency - (warmup_steps + active_steps) such that the effective cycle length == profiling_frequency
    max_steps: int = -1, # Max number of training steps (in units of batches) per epoch. -1 means no max_steps, otherwise training loop breaks after `max_steps` each epoch.
):
    """
    Train a model with FSDP and QLoRA/QDoRA.

    Args:

        world_size: Number of GPUs to use. -1 = all available GPUs.
        train_type: "full", "lora", "qlora", or "custom_qlora"
        llama_pro_path: Path to the quantized llama pro model
        batch_size: Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
        context_length: Max length of input sequence (in tokens)
        gradient_accumulation_steps: How many steps to accumulate gradients over (increases effective batch size)
        num_epochs: How many epochs of training to do
        dataset: alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
        dataset_samples: Number of samples in an epoch if using "alpaca_sample" or "dummy" dataset
        sharding_strategy: Sharding strategy for FSDP
        use_gradient_checkpointing: Use FSDP's activation checkpointing
        reentrant_checkpointing: Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
        use_cpu_offload: Use FSDP's CPU offloading
        use_activation_cpu_offload: Use FSDP's activation CPU offloading
        low_memory: Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
        no_sync: Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
        precision: Training precision. autocast precisions use mixed precision
        model_name: Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        save_model: Save the resulting model
        output_dir: Output directory to save the final model to
        lora_rank: LoRA rank for lora/qlora
        lora_alpha: LoRA alpha for lora/qlora
        lora_dropout: LoRA dropout for lora/qlora
        lora_target_modules: If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
        verbose: Whether to print extra info for debugging
        lr: Learning rate
        apply_gradient_clipping: Apply gradient norm clipping
        grad_norm: Gradient norm clipping
        wd: Weight decay
        profile_memory: Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
        optimizer: Optimizer. PyTorch 2.4 nightly adds CPU fused Adam/AdamW which should improve offload training speed.
        lr_scheduler: Learning Rate Scheduler. linear and cosine warm up for 10% of training steps.
        loading_workers: Number of layers to load and quantize in parallel per GPU. Default of -1 uses heuristics to set worker count.
        log_to: Where to log output
        master_addr: For distributed training
        master_port: For distributed training, must be the same for all processes
        seed: Random seed
        project_name: For wandb logging
        name: For wandb logging
        group: For wandb logging
        entity: For wandb logging
        n_bits: passed to hqq
        profiling_output: Output file for profiling
    """

    # Set world size
    if world_size == -1:
        world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
+    local_rank = int(os.environ["LOCAL_RANK"])

    # Get all args which will be passed to fsdp_main
    args = dict(locals())
    set_seed(args['seed'])
    validate_args(args)
    if args['verbose']: print(args)

    # If lora_target_modules is 'all', set sensible defaults for llama + mistral type modules
    # See peft.utils.constants -> TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING for the current defaults
    if lora_target_modules == "all":
        args["lora_target_modules"] = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
    elif lora_target_modules.lower() == "default":
        args["lora_target_modules"] = None

    if args["precision"] in ["bf16", "bf16_autocast", "bf16_buffers_autocast"] and not torch.cuda.is_bf16_supported():
        raise ValueError('Current device does not support bfloat16')

    # Set no_sync if using cpu_offload and gradient accumulation. Turn off if not using gradient accumulation
    if args["use_cpu_offload"] and args["gradient_accumulation_steps"] > 1:
        args["no_sync"] = True
    elif args["no_sync"] and args["gradient_accumulation_steps"] == 1:
        args["no_sync"] = False

    if args["train_type"] in ["hqq_lora"] and HQQLinear is None:
        raise ValueError("HQQ is required to train with `train_type='hqq_lora'`. See ReadMe for details.")

    if args["optimizer"] in ["fused_adam", "fused_adamw"] and args["use_cpu_offload"] and parse(torch.__version__) < parse("2.4dev"):
        raise ValueError(f"Optimizer '{args['optimizer']}' with `use_cpu_offload=True` requires at least PyTorch 2.4 Nightly with fused Adam/AdamW CPU support.")

    # Run
-    mp.spawn(fsdp_main,
-        args=(world_size, args),
-        nprocs=torch.cuda.device_count(),
-        join=True)
+    fsdp_main(local_rank, world_size, args)

# Entry point, one line wrapper around fsdp_qlora(), use fastcore's call_parse to parse args from command line
@call_parse()
def main(
    world_size: int = -1, # Number of GPUs to use. -1 = all available GPUs.
    train_type: Param("", choices=["full", "lora", "qlora", "custom_qlora", "custom_lora", "hqq_lora", "hqq_dora", "bnb_dora", "bnb_llama_pro", "hqq_llama_pro"]) = "qlora", # "full", "lora", "qlora", or "custom_qlora"
    llama_pro_path: str = None, # Path to the quantized llama pro model
    batch_size: int = 1, # Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
    context_length: int = 512, # Max length of input sequence (in tokens)
    gradient_accumulation_steps: int = 1, # How many steps to accumulate gradients over (increases effective batch size)
    num_epochs: int = 1, # How many epochs of training to do
    dataset: Param("", choices=["alpaca", "alpaca_sample", "dummy", "guanaco", "sql", "orca_math"]) = "alpaca_sample", # alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
    dataset_samples: int = 512, # Number of samples in an epoch if using "alpaca_sample" or "dummy" dataset
    sharding_strategy: Param("", choices=["full_shard", "shard_grad_op", "ddp", "hybrid_full_shard", "hybrid_shard_grad_op"]) = "full_shard", # Sharding strategy for FSDP
    use_gradient_checkpointing: bool_arg = True, # Use FSDP's activation checkpointing
    reentrant_checkpointing: bool_arg = False, # Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
    use_cpu_offload: bool_arg = True, # Use FSDP's CPU offloading
    use_activation_cpu_offload: bool_arg = False, # Use FSDP's activation CPU offloading
    low_memory: bool_arg = True, # Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
    no_sync: bool_arg = False, # Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
    precision: Param("", choices=["fp32", "bf16", "fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]) = "bf16", # Training precision. autocast precisions use mixed precision
    model_name: str = "meta-llama/Llama-2-7b-hf", # Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    save_model: bool_arg = False, # Save the resulting model
    output_dir: str = "output", # Output directory to save the final model to
    lora_rank: int = 64, # LoRA rank for lora/qlora
    lora_alpha: int = 16, # LoRA alpha for lora/qlora
    lora_dropout: float = 0.1, # LoRA dropout for lora/qlora
    lora_target_modules: Param("", choices=["all", "default"]) = "all", # If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
    verbose: bool_arg = False, # Whether to print extra info for debugging
    lr: float = 1e-5, # Learning rate
    apply_gradient_clipping: bool_arg = False, # Apply gradient norm clipping
    grad_norm: float = 0.3, # Gradient norm clipping
    wd: float = 0.1, # Weight decay
    profile_memory: bool_arg = False, # Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
    optimizer: Param("", choices=["adamw", "adam", "sgd", "adadelta"]) = "adamw", # Optimizer
    lr_scheduler: Param("", choices=["constant", "linear", "cosine"]) = "constant", # Learning Rate Scheduler. linear and cosine warm up for 10% of training steps.
    loading_workers: int = -1, # Number of layers to load and quantize in parallel per GPU. Default of -1 uses heuristics to set worker count.
    log_to: Param("", choices=["tqdm", "wandb", "stdout"]) = "tqdm", # Where to log output
    master_addr: str = "localhost", # For distributed training
    master_port: str = "12355", # For distributed training, must be the same for all processes
    seed: int = 42, # Random seed
    project_name: str = "fsdp_qlora", # For wandb logging
    name: str = None, # For wandb logging
    group: str = None, # For wandb logging
    entity: str = None, # For wandb logging
    n_bits: int = 4, # passed to hqq
    profile: bool_arg = False, # Whether to profile with torch.profiler
    profiling_output: str = "profiles", # Output file prefix for profiling
    with_stack: bool_arg = False, # Output stacks for profiling. Note that setting export_memory_timeline will automatically export traces since `with_stack` must be true to profile memory.
    with_shapes: bool_arg = False, # Output shapes for profiling. Can impact performance.  Note that setting export_memory_timeline will automatically export traces since `with_shapes` must be true to profile memory.
    export_trace: bool_arg = True, # Output trace for profiling
    export_memory_timeline: bool_arg = False, # Output memory timelinefor profiling
    wait_steps: int = 0, # Wait steps when running profiler.  Only used if repeat != 0.
    warmup_steps: int = 1, # Warmup steps when running profiler
    active_steps: int = 2,  # Active steps when running profiler
    repeat: int = 0, #Number of profiler cycles (wait + warmup + active) if > 0, else repeats forever
    profiling_frequency: int = 10, # Profiling frequency in steps.  Only used if repeat == 0, in which case wait_steps will be set to profiling_frequency - (warmup_steps + active_steps) such that the effective cycle length == profiling_frequency
    max_steps: int = -1, # Max number of training steps (in units of batches) per epoch. -1 means no max_steps, otherwise training loop breaks after `max_steps` each epoch.
):
    fsdp_qlora(**locals())
```

:::

## kubernetes の job マニフェストを作成

```yaml: job-manifest.yaml
# Service configuration for multinode.
apiVersion: v1
kind: Service
metadata:
  name: multinode-svc
spec:
  clusterIP: None  # ClusterIP set to None for headless service.
  ports:
  - name: nccl  # Port for torchrun master-worker node communication.
    port: 29500
    targetPort: 29500
  selector:
    job-name: multinode-job  # Selector for pods associated with this service.

---

apiVersion: batch/v1
kind: Job
metadata:
  name: multinode-job
spec:
  completionMode: Indexed
  completions: 2
  parallelism: 2
  template:
    spec:
      restartPolicy: Never
      subdomain: multinode-svc  # Subdomain for the headless service.
      containers:
      - image: 「your image」
        name: multinode
        env:
        - name: MASTER_ADDR
          value: multinode-job-0.multinode-svc.default.svc.cluster.local  # Node with rank 0 is chosen as the master node.
        - name: MASTER_PORT
          value: '29500'
        - name: NNODES
          value: '2'  # Number of training nodes.
        - name: NGPUS
          value: '1'  # Number of GPUs in the machine.
        - name: NCCL_DEBUG
          value: 'ERROR'  # Debug level set to ERROR for production
        ports:
        - containerPort: 29500
          name: nccl
        command: ["sh", "-c", "torchrun --nnodes=$NNODES --node_rank=$JOB_COMPLETION_INDEX --nproc_per_node=$NGPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT train.py --world_size=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --model_name meta-llama/Meta-Llama-3-70B --batch_size 2 --context_length 512 --precision bf16 --train_type qlora --use_gradient_checkpointing true --use_cpu_offload true --dataset alpaca --reentrant_checkpointing true --verbose True"]
        resources:
          limits:
            nvidia.com/gpu: 1

```

上記のマニフェストを apply すると、70B の Llama3 で学習ができた！
後は、データセットを自前のものに拡張したり、学習プロセスを変更したり、煮るなり焼くなりしてもらえばいいと思います。

# 4. 注意点

- 学習が実現できたのはいいものの、FSDP では大量のパラメータ等をやりとりするので、ネットワークがボトルネックになってかなり学習速度が落ちた。試しに 7B の Llama3 でシングルノードとマルチノードでの学習速度を比較すると、30~40 倍くらいの学習時間がかかっていることがわかった()
- そもそも、fsdp_qlora の公式ではシングルノードに GPU を二台搭載した計算機を想定してパフォーマンス測定などがされており、マルチノードはそんなに推奨されてないのかも（マルチノード学習に関しては、サンプルスクリプトがあるだけで詳しい説明がない）
  ネットワーク環境周りをもうちょっと見直せば多少は早くなるのかな？何憶というパラメータをやりとりしてるので、これだけ遅くなるのも妥当な気もするが...
- 対応策としては、sharding_strategy を変えると、メモリ使用量と引き換えにノード間の通信回数を減らせたりする。hybrid_full_shard を選択すると、かなり学習速度の改善が見られた（体感シングルノードの時とあまり変わらないくらい）が、当初の目的だった 24GB GPU×2 ではメモリが足りなくなる。
- 現実的な速度で学習を行いたければ、計算機を増やすか、おとなしく GPU を二台搭載した計算機を用意して、ノード間の通信をなくすしかなさそう。
- 後、学習プロセスで generate 関数などのトークン生成を行いたい場合、これらの関数は FSDP の分割計算がうまく適応できないみたいで、FSDP.summon_full_params を使ってパラメータを一度全部収集する必要があるみたいです。詳しくは以下を参照。

https://github.com/pytorch/pytorch/issues/123962
