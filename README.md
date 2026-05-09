# MABI
Multi-Agent Business Intelligence

## 运行
- 使用 `conda` 创建虚拟环境
```cmd
conda create -n MABI python=3.12 -y
conda activate MABI
pip install -r requirements.txt
```
- 然后在根目录创建 `.env` 文件，填入 `assets/环境变量.png` 的值
> 数据库用的是我租的服务器上的 MySQL</br>
> LLM 调用 DeepSeek 就直接用我的 API_KEY 吧
- 启动
```cmd
streamlit run app.py
```

---

Requirements.md 复制粘贴 Canvas 上的作业要求

项目目录结构按照作业要求

logs/ 是 app 运行时记录的日志