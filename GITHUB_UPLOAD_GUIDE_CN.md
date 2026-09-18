# GitHub 上传与论文链接说明

## 1. 本地检查

在 VS Code 中打开整个仓库文件夹，而不是只打开某个 `.py` 文件。终端执行：

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/verify_manuscript_results.py
```

三个命令都通过后再上传。`verify_manuscript_results.py` 专门检查仓库内的
冻结数据是否与正文数字对应。

## 2. 创建 GitHub 仓库

在 GitHub 新建一个空仓库，例如 `lp-lasso-4vfb-experiments`。不要让 GitHub
自动添加 README、`.gitignore` 或 license，因为资源包中已经包含前两者；
license 可在确认作者和期刊要求后单独选择。

在本地仓库根目录执行：

```bash
git init
git add .
git commit -m "Release reproducibility code and paper results"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/lp-lasso-4vfb-experiments.git
git push -u origin main
```

若仓库已经存在，不要再次 `git init`；把本资源包逐项复制进去，检查
`git status` 后正常 commit/push 即可。

## 3. GitHub 页面检查

确认以下内容可以直接看到：

- `README.md` 能正常渲染；
- `src/lp_lasso_4vfb/` 下的模块完整；
- `results/reference_20260907T164315Z/` 和 `results/scale_validation/` 已上传；
- `paper_assets/` 中有 `.tex`、`.pdf` 和 `.png`；
- `notebooks/colab_runner.ipynb` 可以用 “Open in Colab” 打开。

然后编辑 Colab notebook 第一段代码中的 `REPO_URL`，换成实际 GitHub URL，
再提交一次。这一步保证读者点击 notebook 后能直接克隆正确仓库。

## 4. 论文中的链接

建议不要只链接默认分支，因为以后修改 `main` 会改变可复现实验版本。
完成最终检查后创建 release/tag，例如：

```bash
git tag -a v1.0-paper -m "Code and data for the submitted manuscript"
git push origin v1.0-paper
```

正文可写：

```latex
The code, fixed random seeds, raw logs, and scripts used to reproduce the
reported tables and figures are available at
\url{https://github.com/YOUR_ACCOUNT/lp-lasso-4vfb-experiments/tree/v1.0-paper}.
```

如果期刊实行双盲审稿，应先使用期刊允许的匿名代码仓库；接收后再替换为
公开 GitHub release。长期归档时，可把 GitHub release 同步到 Zenodo，并在
最终稿中优先给 DOI。

## 5. 哪些结果应当提交

资源包已经包含正文实际使用的冻结结果。`results/runs/` 是你以后本地或
Colab 重跑产生的输出，默认被 `.gitignore` 排除。只有在你决定用新运行替换
正文数字时，才应把新目录移入一个明确命名的冻结目录，并重新运行审计与
生成资产脚本。

