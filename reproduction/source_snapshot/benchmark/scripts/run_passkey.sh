MODEL=Meta-Llama-3.1-8B-Instruct
# 模型路径
# MODELPATH=/workspace/models/Meta-Llama-3.1-8B-Instruct
MODELPATH=/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct

OUTPUT_DIR=results_passkey/$MODEL

mkdir -p $OUTPUT_DIR

#  --iterations 100：每个长度测试几次。每次 pass key 和插入深度不同。
# --fixed-length 4000：脚本里的“字符级垃圾文本长度基准”。注意源码会做 args.fixed_length = args.fixed_length * 4，所以它不是最终 token 数。实际 token 数看输出里的 Prompt is xxxx tokens。
length=13000

# --algo-config-path：指定 Twilight 用哪种稀疏注意力配置。
algo_cfg_path=./configs/config_quest_twi.json

python3 ./passkey/passkey.py -m $MODELPATH \
    --iterations 10 --fixed-length $length \
    --algo-config-path $algo_cfg_path \
    --output-dir $OUTPUT_DIR