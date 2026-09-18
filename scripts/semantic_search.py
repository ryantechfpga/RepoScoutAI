#!/usr/bin/env python3
"""
语义检索智能体：
  1. LLM 解析自然语言描述 -> 关键词 JSON
  2. GitHub Search API 检索仓库
  3. 拉取每个仓库的 README
  4. LLM 生成项目介绍
  5. 输出 Markdown / JSON 报告
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------- 配置 ----------------
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
AI_API_KEY   = os.environ.get("AI_API_KEY", "").strip()
AI_BASE_URL  = (os.environ.get("AI_BASE_URL") or "https://api.openai.com/v1").strip()
AI_MODEL     = (os.environ.get("AI_MODEL") or "gpt-4o-mini").strip()

GH_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    GH_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


# ---------------- LLM 调用 ----------------
def llm_chat(system: str, user: str, temperature: float = 0.2) -> str:
    """调用兼容 OpenAI 格式的 Chat Completions API。"""
    url = f"{AI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }

    for attempt in range(3):
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        
        # ============ 新增：打印详细错误 ============
        print(f"[llm] 状态码: {r.status_code}", file=sys.stderr)
        print(f"[llm] 响应头 X-RateLimit: "
              f"{r.headers.get('X-RateLimit-Remaining')} / "
              f"{r.headers.get('X-RateLimit-Limit')}", file=sys.stderr)
        print(f"[llm] 响应正文: {r.text[:800]}", file=sys.stderr)
        # ==========================================
        
        if r.status_code in (429, 500, 502, 503):
            wait = 5 * (attempt + 1)
            print(f"[warn] LLM {r.status_code}, 等待 {wait}s 重试...", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()

    raise RuntimeError("LLM 调用失败（重试耗尽）")


def llm_json(system: str, user: str):
    """调用 LLM 并解析 JSON，容错处理常见的 Markdown 包裹。"""
    text = llm_chat(system, user).strip()
    # 去掉可能的 ```json ... ```
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    # 提取第一个 JSON 数组/对象
    m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if m:
        text = m.group(1)
    return json.loads(text)


# ---------------- 步骤 1：解析描述 ----------------
def extract_keywords(description: str) -> dict:
    system = (
        "你是一个开源项目检索助手。用户会用自然语言描述他想找的开源项目。\n"
        "请完成两件事：\n"
        "1) 提取 3-6 个最适合用于 GitHub 仓库搜索的英文关键词/短语（技术栈、功能、领域）\n"
        "2) 用一句话总结用户的核心需求\n"
        "返回严格的 JSON，格式如下，不要任何额外文字：\n"
        '{"keywords": ["vector database", "embedding", "python"], '
        '"summary": "用户想找一个轻量级、支持中文的向量数据库"}'
    )
    user = f"用户描述：{description}"
    data = llm_json(system, user)

    kws = data.get("keywords") or []
    if not isinstance(kws, list) or not kws:
        raise RuntimeError(f"关键词解析失败：{data}")

    return {"keywords": kws, "summary": data.get("summary", "")}


# ---------------- 步骤 2：GitHub 搜索 ----------------
# def search_repos(keywords, language, min_stars, per_page=50) -> list[dict]:
#     q_parts = [" ".join(keywords)]
#     if language:
#         q_parts.append(f"language:{language}")
#     if min_stars:
#         q_parts.append(f"stars:>={min_stars}")
#     query = " ".join(q_parts)

#     print(f"[search] q = {query!r}")

#     r = requests.get(
#         "https://api.github.com/search/repositories",
#         headers=GH_HEADERS,
#         params={"q": query, "per_page": per_page, "sort": "stars"},
#         timeout=30,
#     )
#     if r.status_code == 422:
#         print(f"[warn] 查询语法非法，回退到纯关键词", file=sys.stderr)
#         r = requests.get(
#             "https://api.github.com/search/repositories",
#             headers=GH_HEADERS,
#             params={"q": " ".join(keywords), "per_page": per_page, "sort": "stars"},
#             timeout=30,
#         )
#     r.raise_for_status()
#     return r.json().get("items", [])

def search_repos(keywords, language, min_stars, per_page=50) -> list[dict]:
    """
    构造 GitHub 搜索查询并调用 API。
    多语言用空格分隔（language:verilog language:systemverilog），
    而不是逗号（language:verilog,systemverilog 是无效语法）。
    """
    q_parts = [" ".join(keywords)]

    # 多语言支持：按空格或逗号拆分，逐个添加 language: 限定符
    if language:
        langs = re.split(r"[\s,]+", language.strip())
        for lang in langs:
            if lang:
                q_parts.append(f"language:{lang}")

    if min_stars:
        q_parts.append(f"stars:>={min_stars}")

    query = " ".join(q_parts)

    # ============ 日志：打印完整查询 ============
    print("=" * 60)
    print(f"[search] 关键词        : {keywords}")
    print(f"[search] 语言限定      : {language!r}")
    print(f"[search] 最低 Star     : {min_stars!r}")
    print(f"[search] 最终查询字符串: {query!r}")
    print(f"[search] 请求 URL      : https://api.github.com/search/repositories")
    print(f"[search] 请求参数      : q={query!r}, per_page={per_page}, sort=stars")
    print("=" * 60)
    # ============================================

    def _do_search(q: str) -> requests.Response:
        return requests.get(
            "https://api.github.com/search/repositories",
            headers=GH_HEADERS,
            params={"q": q, "per_page": per_page, "sort": "stars"},
            timeout=30,
        )

    r = _do_search(query)

    # 打印响应状态和速率限制
    print(f"[search] 响应状态码    : {r.status_code}")
    print(f"[search] 速率限制剩余  : {r.headers.get('X-RateLimit-Remaining')}"
          f" / {r.headers.get('X-RateLimit-Limit')}")
    print(f"[search] 速率重置时间  : {r.headers.get('X-RateLimit-Reset')}")

    # 422：语法非法，回退到纯关键词
    if r.status_code == 422:
        print(f"[warn] 422 查询语法非法，响应内容: {r.text[:500]}", file=sys.stderr)
        fallback_q = " ".join(keywords)
        print(f"[warn] 回退查询字符串: {fallback_q!r}", file=sys.stderr)
        r = _do_search(fallback_q)

    # 403 / 429：速率限制
    if r.status_code in (403, 429):
        print(f"[warn] {r.status_code} 可能触发速率限制，响应内容: {r.text[:500]}",
              file=sys.stderr)

    r.raise_for_status()

    data = r.json()
    items = data.get("items", [])

    # 打印结果数量和总数
    print(f"[search] 本次返回数量  : {len(items)}")
    print(f"[search] GitHub 报告总数: {data.get('total_count', 'N/A')}")
    if data.get("incomplete_results"):
        print(f"[warn] GitHub 报告结果不完整 (incomplete_results=true)", file=sys.stderr)

    # 如果返回 0 条，打印前几个字段帮助排查
    if not items:
        print(f"[warn] 查询返回 0 条结果，请检查关键词和限定条件是否过严",
              file=sys.stderr)

    return items
# ---------------- 步骤 3：获取 README ----------------
def fetch_readme(full_name: str, max_chars: int = 3500) -> str:
    url = f"https://api.github.com/repos/{full_name}/readme"
    headers = dict(GH_HEADERS)
    headers["Accept"] = "application/vnd.github.raw+json"

    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 404:
            return ""
        r.raise_for_status()
        text = r.text
    except Exception as e:
        print(f"[warn] 读取 README 失败 {full_name}: {e}", file=sys.stderr)
        return ""

    # 去掉 HTML 标签、Badge、多余空行
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars]


# ---------------- 步骤 4：生成项目介绍 ----------------
def summarize_repo(repo: dict, readme: str) -> str:
    system = (
        "你是一个技术摘要助手。给定一个 GitHub 仓库的信息和 README 片段，\n"
        "请用 2-3 句中文概括：① 这个项目做什么；② 核心功能；③ 适用场景。\n"
        "只输出概括文字，不要标题、不要 Markdown 格式、不要客套话。"
    )
    user = (
        f"仓库名：{repo['full_name']}\n"
        f"官方描述：{repo.get('description') or '（无）'}\n"
        f"主要语言：{repo.get('language') or '未知'}\n"
        f"Star 数：{repo.get('stargazers_count')}\n"
        f"Topics：{', '.join(repo.get('topics') or []) or '（无）'}\n\n"
        f"README 片段：\n{readme or '（README 为空）'}"
    )
    try:
        return llm_chat(system, user, temperature=0.3).strip()
    except Exception as e:
        print(f"[warn] 生成介绍失败 {repo['full_name']}: {e}", file=sys.stderr)
        return repo.get("description") or "（无法生成介绍）"


# ---------------- 主流程 ----------------
def main() -> int:
    description = os.environ.get("INPUT_DESCRIPTION", "").strip()
    language    = os.environ.get("INPUT_LANGUAGE", "").strip()
    min_stars   = os.environ.get("INPUT_MIN_STARS", "").strip()

    try:
        top_n = int(os.environ.get("INPUT_TOP_N", "8") or 8)
    except ValueError:
        top_n = 8

    outdir = Path(os.environ.get("OUTPUT_DIR", "results"))
    outdir.mkdir(parents=True, exist_ok=True)

    if not description:
        print("::error::请提供描述")
        return 1
    if not AI_API_KEY:
        print("::error::缺少 AI_API_KEY，请在仓库 Secrets 中配置")
        return 1

    # --- 1) 解析描述 ---
    print("[1/5] 调用 LLM 解析描述...")
    parsed = extract_keywords(description)
    keywords = parsed["keywords"]
    summary = parsed["summary"]
    print(f"       关键词: {keywords}")
    print(f"       摘要  : {summary}")

    # --- 2) 搜索仓库 ---
    print("[2/5] 检索 GitHub 仓库...")
    candidates = search_repos(keywords, language, min_stars)
    print(f"       候选池: {len(candidates)} 个")

    # 按 star 排序取前 N（也可换成更复杂的打分）
    candidates.sort(key=lambda r: r.get("stargazers_count") or 0, reverse=True)
    top = candidates[:top_n]

    # --- 3) & 4) 拉 README + 生成介绍 ---
    print(f"[3/5] 获取 README 并生成介绍（共 {len(top)} 个）...")
    enriched = []
    for i, repo in enumerate(top, 1):
        print(f"   [{i}/{len(top)}] {repo['full_name']}")
        readme = fetch_readme(repo["full_name"])
        intro  = summarize_repo(repo, readme)
        enriched.append({"repo": repo, "intro": intro})
        time.sleep(1)  # 避免过于频繁

    # --- 5) 输出报告 ---
    print("[5/5] 生成报告...")
    lines = []
    lines.append("## 🔎 语义检索结果\n")
    lines.append(f"**你的描述**：{description}\n")
    lines.append(f"**AI 理解**：{summary}\n")
    lines.append(f"**提取关键词**：{', '.join(f'`{k}`' for k in keywords)}\n")
    if language:
        lines.append(f"**语言限定**：`{language}`\n")
    if min_stars:
        lines.append(f"**最低 Star**：`{min_stars}`\n")
    lines.append(f"**候选池**：{len(candidates)} 个仓库，展示前 {len(enriched)} 个\n")
    lines.append("---\n")

    for i, item in enumerate(enriched, 1):
        repo  = item["repo"]
        intro = item["intro"]
        stars = repo.get("stargazers_count") or 0
        stars_s = f"{stars/1000:.1f}k" if stars >= 1000 else str(stars)

        lines.append(f"### {i}. [{repo['full_name']}]({repo['html_url']})\n")
        lines.append(
            f"⭐ **{stars_s}** ｜ 🖥 `{repo.get('language') or '-'}` "
            f"｜ 📅 {repo.get('pushed_at','')[:10]} "
            f"｜ 📜 `{(repo.get('license') or {}).get('spdx_id') or '无协议'}`\n"
        )
        lines.append(f"**项目介绍**：{intro}\n")

        topics = repo.get("topics") or []
        if topics:
            lines.append("**Topics**：" + ", ".join(f"`{t}`" for t in topics[:8]) + "\n")

        lines.append("<details><summary>原始描述</summary>\n")
        lines.append(f"\n> {repo.get('description') or '（无）'}\n")
        lines.append("\n</details>\n")
        lines.append("---\n")

    markdown = "\n".join(lines)
    (outdir / "report.md").write_text(markdown, encoding="utf-8")

    # JSON 备份
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {"description": description, "language": language, "min_stars": min_stars},
        "parsed": parsed,
        "results": [
            {
                "rank": i,
                "full_name": item["repo"]["full_name"],
                "url": item["repo"]["html_url"],
                "stars": item["repo"].get("stargazers_count"),
                "language": item["repo"].get("language"),
                "description": item["repo"].get("description"),
                "ai_intro": item["intro"],
            }
            for i, item in enumerate(enriched, 1)
        ],
    }
    (outdir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Actions Summary
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(markdown)

    print("\n" + markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())