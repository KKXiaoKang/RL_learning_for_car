#!/bin/bash

echo "===================================================="
echo "           Ubuntu Anaconda 检查工具"
echo "===================================================="
echo

TOTAL_CHECKS=0
SUCCESS_CHECKS=0
FOUND_ANACONDA=0
PATHS_LIST=()

# 1. 检查常见的Anaconda安装路径
echo "[1] 检查常见的Anaconda安装路径..."
echo

COMMON_PATHS=(
    "$HOME/anaconda3"
    "$HOME/anaconda2"
    "$HOME/miniconda3"
    "$HOME/miniconda2"
    "$HOME/opt/anaconda3"
    "$HOME/opt/anaconda2"
    "/opt/anaconda3"
    "/opt/anaconda2"
    "/usr/local/anaconda3"
    "/usr/local/anaconda2"
)

for path in "${COMMON_PATHS[@]}"; do
    ((TOTAL_CHECKS++))
    echo "检查路径: $path"
    if [ -d "$path" ]; then
        echo "   ✓ 目录存在"
        ((SUCCESS_CHECKS++))
        FOUND_ANACONDA=1
        PATHS_LIST+=("$path")
        
        # 检查是否是有效的Anaconda目录
        if [ -f "$path/bin/conda" ]; then
            echo "   ✓ 确认是有效的Anaconda安装"
        fi
    else
        echo "   ✗ 目录不存在"
    fi
    echo
done

# 2. 检查conda命令是否在PATH中
echo "[2] 检查conda命令可用性..."
echo

((TOTAL_CHECKS++))
echo "检查conda命令..."
if command -v conda &> /dev/null; then
    echo "   ✓ conda命令可用"
    ((SUCCESS_CHECKS++))
    FOUND_ANACONDA=1
    
    # 获取conda版本信息
    conda_version=$(conda --version 2>/dev/null)
    echo "   版本: $conda_version"
    
    # 获取conda安装路径
    conda_path=$(which conda)
    echo "   安装路径: $conda_path"
    PATHS_LIST+=("$conda_path")
    
    # 获取conda信息
    echo
    echo "   Conda环境信息:"
    conda info --envs 2>/dev/null | grep -v "^#" | grep -v "^$" | while read -r env; do
        echo "     环境: $env"
    done
else
    echo "   ✗ conda命令不可用"
fi
echo

# 3. 检查python命令（可能来自Anaconda）
echo "[3] 检查Python安装..."
echo

((TOTAL_CHECKS++))
echo "检查python命令..."
if command -v python &> /dev/null; then
    echo "   ✓ python命令可用"
    ((SUCCESS_CHECKS++))
    
    python_version=$(python --version 2>&1)
    echo "   版本: $python_version"
    
    # 检查python是否来自Anaconda
    python_path=$(which python)
    if [[ "$python_path" == *"anaconda"* ]] || [[ "$python_path" == *"conda"* ]]; then
        echo "   ✓ Python来自Anaconda安装: $python_path"
        FOUND_ANACONDA=1
        PATHS_LIST+=("$python_path")
    else
        echo "   ✗ Python不是来自Anaconda: $python_path"
    fi
else
    echo "   ✗ python命令不可用"
fi
echo

# 4. 检查环境变量
echo "[4] 检查环境变量..."
echo

((TOTAL_CHECKS++))
echo "检查环境变量..."
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "   ✓ CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
    ((SUCCESS_CHECKS++))
    FOUND_ANACONDA=1
else
    echo "   ✗ CONDA_DEFAULT_ENV未设置"
fi

if [ -n "$CONDA_PREFIX" ]; then
    echo "   ✓ CONDA_PREFIX: $CONDA_PREFIX"
    ((SUCCESS_CHECKS++))
    FOUND_ANACONDA=1
    PATHS_LIST+=("$CONDA_PREFIX")
else
    echo "   ✗ CONDA_PREFIX未设置"
fi
echo

# 5. 检查.bashrc、.zshrc等配置文件
echo "[5] 检查Shell配置文件..."
echo

SHELL_FILES=(
    "$HOME/.bashrc"
    "$HOME/.bash_profile"
    "$HOME/.zshrc"
    "$HOME/.profile"
)

for file in "${SHELL_FILES[@]}"; do
    ((TOTAL_CHECKS++))
    echo "检查文件: $file"
    if [ -f "$file" ]; then
        if grep -q "anaconda\|conda" "$file" 2>/dev/null; then
            echo "   ✓ 文件中包含anaconda/conda配置"
            ((SUCCESS_CHECKS++))
            FOUND_ANACONDA=1
            PATHS_LIST+=("$file中的配置")
        else
            echo "   ✗ 文件中不包含anaconda/conda配置"
        fi
    else
        echo "   ✗ 文件不存在"
    fi
    echo
done

# 6. 检查conda的配置文件
echo "[6] 检查Conda配置文件..."
echo

CONDA_CONFIG_FILES=(
    "$HOME/.condarc"
    "/etc/conda/condarc"
)

for file in "${CONDA_CONFIG_FILES[@]}"; do
    ((TOTAL_CHECKS++))
    echo "检查文件: $file"
    if [ -f "$file" ]; then
        echo "   ✓ Conda配置文件存在"
        ((SUCCESS_CHECKS++))
        FOUND_ANACONDA=1
        PATHS_LIST+=("$file")
    else
        echo "   ✗ Conda配置文件不存在"
    fi
    echo
done

# 显示总体检查结果
echo "===================================================="
echo "                总体检查结果"
echo "===================================================="
echo

echo "执行的检查总数: $TOTAL_CHECKS"
echo "成功的检查项: $SUCCESS_CHECKS"
if [ $TOTAL_CHECKS -gt 0 ]; then
    SUCCESS_RATE=$((SUCCESS_CHECKS * 100 / TOTAL_CHECKS))
    echo "检查成功率: ${SUCCESS_RATE}%"
else
    echo "检查成功率: 0%"
fi
echo

if [ $FOUND_ANACONDA -eq 1 ]; then
    echo "✓ 检查结果: 系统中已安装 Anaconda 或相关组件"
    echo
    echo "发现的安装路径和信息:"
    for path in "${PATHS_LIST[@]}"; do
        echo "  - $path"
    done | sort -u
else
    echo "✗ 检查结果: 系统中未找到 Anaconda"
    echo
    echo "可能的原因:"
    echo "  - Anaconda 未安装"
    echo "  - 安装在非标准路径"
    echo "  - 安装后未正确配置环境变量"
    echo "  - 需要重新登录或执行 source ~/.bashrc"
fi

echo
echo "===================================================="
echo "                检查完成"
echo "===================================================="
echo

# 显示系统信息
echo "系统信息:"
echo "  用户名: $(whoami)"
echo "  主机名: $(hostname)"
echo "  系统: $(lsb_release -ds 2>/dev/null || echo "Unknown")"
echo "  架构: $(uname -m)"
echo "  Shell: $SHELL"
echo
