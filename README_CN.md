# 📄 pdf-mate

<p align="center">
  <strong>AI 驱动的 PDF 解析、智能摘要与对话式问答工具</strong>
</p>

<p align="center">
  <a href="https://github.com/junjunup/pdf-mate/stargazers"><img src="https://img.shields.io/github/stars/junjunup/pdf-mate?style=social" alt="Stars"></a>
  <a href="https://github.com/junjunup/pdf-mate/releases"><img src="https://img.shields.io/github/v/release/junjunup/pdf-mate" alt="Release"></a>
  <a href="https://github.com/junjunup/pdf-mate/blob/main/LICENSE"><img src="https://img.shields.io/github/license/junjunup/pdf-mate" alt="License"></a>
  <a href="./README.md"><img src="https://img.shields.io/badge/lang-English-blue" alt="English"></a>
</p>

---

**pdf-mate** 是一个轻量级 Python 工具包，集成了 PDF 内容提取、OCR 识别、RAG 检索增强问答和智能文档摘要等功能。支持多种大语言模型后端（OpenAI、Ollama 等），同时提供强大的命令行工具和简洁的 Gradio Web 界面。

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| **PDF 解析** | 从 PDF 文件中提取文本、表格和图片 |
| **OCR 识别** | 支持扫描件文字识别（基于 Tesseract），支持中英文混合 |
| **RAG 问答** | 基于检索增强生成技术，对文档进行智能问答 |
| **AI 摘要** | 自动生成简洁摘要、详细摘要或要点列表 |
| **多模型支持** | 兼容 OpenAI、Ollama 及任何 OpenAI 兼容 API |
| **本地向量化** | 使用 sentence-transformers 本地生成向量，无需调用 API，保护数据隐私 |
| **CLI 命令行** | 面向高级用户的快速命令行接口 |
| **Web 界面** | 基于 Gradio 的交互式可视化界面 |

## 🚀 快速开始

### 安装

核心包非常轻量（约 100 MB），重型可选依赖按功能拆分为独立模块：

```bash
# 仅安装核心功能（PDF 解析 + CLI，无 API 依赖）
pip install pdf-mate

# 添加 LLM 支持（OpenAI API + 文档摘要）
pip install pdf-mate[llm]

# 添加 RAG 支持（LLM + ChromaDB 向量存储 + 问答）
pip install pdf-mate[rag]

# 添加本地向量化支持（约 2 GB，包含 sentence-transformers）
pip install pdf-mate[local]

# 添加 Web 界面（包含 Gradio）
pip install pdf-mate[web]

# 添加 OCR 支持（需要系统安装 Tesseract）
pip install pdf-mate[ocr]

# 安装全部功能
pip install pdf-mate[all]

# 开发环境
pip install -e ".[dev]"
```

> **提示：** 核心包仅包含 PDF 解析和 CLI 功能，不会引入 OpenAI 或 ChromaDB 依赖。安装 `pdf-mate[rag]` 可获得完整的 RAG 问答体验，安装 `pdf-mate[llm]` 可获得文档摘要功能。

### 命令行使用

```bash
# 解析 PDF 文件
pdf-mate parse document.pdf

# 解析并保存为 Markdown 格式
pdf-mate parse document.pdf -o output.md --format markdown

# 生成 AI 摘要
pdf-mate summary document.pdf

# 交互式问答模式
pdf-mate ask document.pdf

# 直接提问
pdf-mate ask document.pdf -q "这篇文档的主要观点是什么？"

# OCR 识别扫描件（支持中英文）
pdf-mate ocr scanned.pdf --lang eng+chi_sim

# 启动 Web 界面
pdf-mate web
```

### Web 界面

```bash
pdf-mate web --port 7860
```

在浏览器中打开 `http://localhost:7860`，即可使用可视化界面：

- **解析标签页** — 上传 PDF 并提取内容
- **摘要标签页** — 生成 AI 文档摘要
- **问答标签页** — 索引文档并进行交互式提问

## 📖 使用示例

### Python API

```python
from pdf_mate.parser import PDFParser

# 解析 PDF
parser = PDFParser()
content = parser.parse("document.pdf")

print(f"页数: {content.page_count}")
print(f"文本: {content.full_text[:500]}")
print(f"表格数量: {len(content.tables)}")
```

### RAG 检索问答

```python
from pdf_mate.parser import PDFParser
from pdf_mate.rag import RAGEngine, RAGConfig

# 配置 RAG 引擎（使用本地向量化）
config = RAGConfig(
    embedding_provider="local",  # 需要: pip install pdf-mate[local]
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gpt-4o-mini",
)
engine = RAGEngine(config)

# 索引文档
parser = PDFParser()
content = parser.parse("research_paper.pdf")
engine.index_document(content.full_text, source_name="research_paper.pdf")

# 提问
answer = engine.query("这篇论文的主要发现是什么？")
print(answer.answer)
print("来源:", [s[0] for s in answer.sources])
```

### 文档摘要

```python
from pdf_mate.parser import PDFParser
from pdf_mate.summary import DocumentSummarizer, SummaryConfig

# 解析并摘要
parser = PDFParser()
content = parser.parse("report.pdf")

config = SummaryConfig(
    llm_model="gpt-4o-mini",
    style="bullets",  # 可选: "concise"(简洁), "detailed"(详细), "bullets"(要点)
)
summarizer = DocumentSummarizer(config)
summary = summarizer.summarize(
    text=content.full_text,
    filename=content.filename,
    page_count=content.page_count,
)

print(f"标题: {summary.title}")
print(f"摘要: {summary.summary}")
print("核心要点:")
for point in summary.key_points:
    print(f"  - {point}")
```

### OCR 文字识别

```python
from pdf_mate.ocr import OCREngine

# 需要: pip install pdf-mate[ocr]
engine = OCREngine(language="eng+chi_sim")  # 英文+简体中文
results = engine.extract_text_from_pdf("scanned.pdf")

for page_num, text in results:
    print(f"--- 第 {page_num + 1} 页 ---")
    print(text)
```

### 异常处理

pdf-mate 提供了统一的异常体系，所有异常均继承自 `PDFMateError`，方便精确捕获和统一处理：

```python
from pdf_mate import PDFParser, PDFMateError, ParseError, LLMError

try:
    parser = PDFParser()
    content = parser.parse("document.pdf")
except ParseError as e:
    print(f"解析失败: {e}")
except PDFMateError as e:
    print(f"pdf-mate 错误: {e}")
```

**异常类型一览：**

| 异常类 | 触发场景 |
|--------|----------|
| `PDFMateError` | 所有异常的基类 |
| `ParseError` | PDF 解析失败（文件不存在、格式损坏等） |
| `OCRError` | OCR 识别失败（Tesseract 未安装、识别错误等） |
| `LLMError` | LLM 调用失败（API 错误、超时等） |
| `EmbeddingError` | 向量化失败 |
| `StorageError` | 向量存储操作失败 |
| `RAGError` | RAG 流程失败 |
| `ConfigError` | 配置错误 |

## 🔧 配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `OPENAI_BASE_URL` | 自定义 API 地址（用于兼容 API 或代理） | - |

### 使用 Ollama（本地大模型）

无需 API 密钥，使用本地运行的开源模型：

```bash
# 启动 Ollama 服务
ollama serve

# 下载模型
ollama pull llama3

# 配合 pdf-mate 使用
pdf-mate ask document.pdf --model llama3 --base-url http://localhost:11434/v1
```

## 🏗️ 项目架构

```
pdf-mate/
├── src/pdf_mate/
│   ├── __init__.py     # 公共 API 导出（懒加载）
│   ├── exceptions.py   # 统一异常层级
│   ├── parser.py       # PDF 文本/表格/图片提取
│   ├── ocr.py          # 扫描件 OCR 识别
│   ├── llm.py          # 多后端 LLM 接口
│   ├── embedding.py    # 文本向量化
│   ├── storage.py      # ChromaDB 向量存储
│   ├── rag.py          # RAG 流程（分块 + 检索 + 生成）
│   ├── summary.py      # 文档智能摘要
│   ├── cli.py          # CLI 命令行（Typer + Rich）
│   └── web.py          # Gradio Web 界面
├── tests/              # pytest 测试套件（178 个测试，89% 覆盖率）
├── pyproject.toml      # 项目配置
└── CHANGELOG.md        # 变更日志
```

### 设计亮点

- **依赖分层**：核心包仅约 100MB，重型依赖按功能拆分为可选 extras，避免安装不需要的包
- **懒加载导入**：`__init__.py` 使用 `__getattr__` + `_LAZY_IMPORTS` 实现延迟导入，加速包加载
- **统一异常体系**：`PDFMateError` 基类 + 7 个语义化子类，异常边界清晰
- **多后端适配**：LLM 和 Embedding 模块均支持多种后端，通过工厂模式创建

## 🛠️ 开发指南

```bash
# 克隆仓库
git clone https://github.com/junjunup/pdf-mate.git
cd pdf-mate

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 运行测试（带覆盖率报告）
pytest --cov=pdf_mate

# 代码检查
ruff check src/ tests/

# 类型检查
mypy src/
```

### 代码质量

| 指标 | 状态 |
|------|------|
| 测试数量 | 178 个（0 失败） |
| 代码覆盖率 | 89% |
| Lint 检查 | ruff 零错误 |
| 类型支持 | PEP 561 (py.typed) |
| 变更日志 | Keep a Changelog 格式 |

## 📋 环境要求

- Python >= 3.10

**核心依赖：**
- `pdfplumber` — PDF 文本和表格提取
- `PyMuPDF` — PDF 图片提取和页面渲染
- `numpy` — 数值计算
- `typer` + `rich` — CLI 框架

**可选依赖：**
- `openai` + `tiktoken` — LLM API 客户端（`pip install pdf-mate[llm]`）
- `chromadb` — RAG 向量数据库（`pip install pdf-mate[rag]`）
- `sentence-transformers` — 本地向量化（`pip install pdf-mate[local]`）
- `gradio` — Web 界面（`pip install pdf-mate[web]`）
- `pytesseract` + `Pillow` — OCR 支持（`pip install pdf-mate[ocr]`）

## 📜 开源协议

本项目基于 MIT 协议开源 — 详见 [LICENSE](LICENSE) 文件。

## 🤝 参与贡献

欢迎提交 Pull Request！参与贡献的流程：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交修改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 发起 Pull Request

---

<p align="center">
  如果 pdf-mate 对你有帮助，请给一个 ⭐ Star 支持！
</p>
