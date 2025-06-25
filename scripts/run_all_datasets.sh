#!/bin/bash

# 定义数据集和对应的 GPU
declare -A DATASET_GPU=(
    ["All-IDB"]=0
    ["AML"]=0
    ["Ascites2020"]=2
    ["Barcelona"]=2
    ["BCFC"]=2
    ["BCI"]=2
    ["BMC"]=3
    ["BMT"]=4
    ["Breast2023"]=4
    ["C_NMC_2019"]=4
    ["CCS-Cell-Cls"]=4
    ["CERVIX93"]=4
    ["CSF2022"]=6
    ["FNAC2019"]=2
    # ["Herlev"]=2
    # ["HiCervix"]=4
    # ["JinWooChoi"]=0
    # ["LDCC"]=3
    # ["MendeleyLBC"]=2
    # ["PS3C"]=2
    # ["Raabin_WBC"]=2
    # ["RepoMedUNM"]=4
    # ["SIPaKMeD"]=5
    # ["Thyroid2024"]=6
    # ["UFSC_OCPap"]=4
)

# 遍历所有数据集，并使用对应的 GPU 训练
for dataset in "${!DATASET_GPU[@]}"; do
    gpu="${DATASET_GPU[$dataset]}"
    echo "Training dataset $dataset on GPU $gpu"
    sh ./scripts/train_simple.sh -b cell_cls configs/cell_cls/quick_test.yaml \
        common.gpu="$gpu" data.dataset="$dataset"
done

echo "All training jobs submitted!"