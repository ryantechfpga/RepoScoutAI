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
        "请完成三件事：\n"
        "1) 提取 4-6 个英文关键词，并给每个词标注优先级\n"
        "   优先级用 1、2、3 表示，数字越小越重要：\n"
        "   1 = 核心词（领域名、核心技术，如 ISP、FPGA）\n"
        "   2 = 重要限定词（接口、平台、协议，如 MIPI、DVP）\n"
        "   3 = 辅助词（场景、形容词，如 camera、embedded）\n"
        "2) 关键词必须是英文，单词或缩写，不要用多词短语\n"
        "3) 用一句话总结用户的核心需求\n"
        "返回严格的 JSON，不要任何额外文字：\n"
        "{\n"
        '  "keywords": [\n'
        '    {"term": "ISP", "priority": 1},\n'
        '    {"term": "FPGA", "priority": 1},\n'
        '    {"term": "MIPI", "priority": 2},\n'
        '    {"term": "DVP", "priority": 2}\n'
        "  ],\n"
        '  "summary": "用户想要一个支持 MIPI/DVP 接口的 ISP FPGA 工程"\n'
        "}"
    )
    user = f"用户描述：{description}"
    data = llm_json(system, user)

    raw_kws = data.get("keywords") or []
    if not isinstance(raw_kws, list) or not raw_kws:
        raise RuntimeError(f"关键词解析失败：{data}")

    # 兼容两种格式：字符串列表 / 对象列表
    keywords = []
    for item in raw_kws:
        if isinstance(item, str):
            keywords.append({"term": item.strip(), "priority": 2})
        elif isinstance(item, dict) and item.get("term"):
            keywords.append({
                "term": str(item["term"]).strip(),
                "priority": int(item.get("priority", 2)),
            })

    if not keywords:
        raise RuntimeError(f"关键词解析失败：{data}")

    return {"keywords": keywords, "summary": data.get("summary", "")}


# ---------------- 步骤 2：GitHub 搜索 ----------------
def search_repos(keywords, language, min_stars, per_page=50,
                 min_results=5) -> list[dict]:
    """
    先用全部关键词检索；结果不足则从最低优先级开始逐个去掉关键词，
    用剩余的高优先级关键词重新检索，直到结果达标或只剩一个关键词。
    """
    # 按优先级升序排序：P1 在前（最重要），P3 在后（最先被去掉）
    # Python 的 sort 是稳定排序，相同优先级的词保持原顺序
    sorted_kws = sorted(keywords, key=lambda k: k["priority"])
    all_terms = [k["term"] for k in sorted_kws]
    total = len(all_terms)

    print("=" * 60)
    print("[search] 关键词按优先级排序：")
    for k in sorted_kws:
        print(f"          P{k['priority']}  {k['term']}")
    print("=" * 60)

    all_results: dict[str, dict] = {}
    tried = set()

    # 从全量开始，每次去掉最后一个（最低优先级）
    for n in range(total, 0, -1):
        current_terms = all_terms[:n]
        query = _build_query(current_terms, language, min_stars)

        if query in tried:
            continue
        tried.add(query)

        round_no = total - n + 1
        print("-" * 60)
        print(f"[search] 第 {round_no} 轮：使用 {n} 个关键词 {current_terms}")
        print(f"[search] 查询字符串: {query!r}")

        status, items = _do_search(query, per_page)

        for it in items:
            fn = it.get("full_name")
            if fn and fn not in all_results:
                all_results[fn] = it

        print(f"[search] 本轮返回 {len(items)} 条，"
              f"累计去重后 {len(all_results)} 个")

        # 达标，停止
        if len(all_results) >= min_results:
            print(f"[search] ✅ 已达到最低数量 {min_results} 个，停止放宽")
            break

        # 只剩一个关键词，仍不足，放弃继续去词
        if n == 1:
            print(f"[search] ⚠️ 已只剩最高优先级关键词，"
                  f"仍不足 {min_results} 个，停止")
            break

        removed = all_terms[n - 1]
        print(f"[search] ❌ 结果不足 {min_results} 个，"
              f"去掉最低优先级关键词 '{removed}'，进入下一轮")

        time.sleep(2)  # 遵守 GitHub 搜索 API 速率限制

    return list(all_results.values())
def _build_query(keywords, language, min_stars) -> str:
    """把关键词列表和限定条件拼成 GitHub 搜索查询字符串。"""
    parts = [" ".join(keywords)]
    if language:
        for lang in re.split(r"[\s,]+", language.strip()):
            if lang:
                parts.append(f"language:{lang}")
    if min_stars:
        parts.append(f"stars:>={min_stars}")
    return " ".join(parts)


def _do_search(query: str, per_page: int) -> tuple[int, list[dict]]:
    """执行一次 GitHub 搜索，返回 (状态码, items)。"""
    print(f"[search] 请求: q={query!r}, per_page={per_page}")
    r = requests.get(
        "https://api.github.com/search/repositories",
        headers=GH_HEADERS,
        params={"q": query, "per_page": per_page, "sort": "stars"},
        timeout=30,
    )
    print(f"[search] 状态码: {r.status_code}, "
          f"速率剩余: {r.headers.get('X-RateLimit-Remaining')}")

    if r.status_code == 200:
        data = r.json()
        items = data.get("items", [])
        print(f"[search] 返回: {len(items)} 条 / 总数 {data.get('total_count')}")
        return 200, items

    print(f"[search] 错误: {r.text[:300]}", file=sys.stderr)
    return r.status_code, []
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
    lines.append(
    "**提取关键词**："
    + "、".join(f"`{k['term']}`(P{k['priority']})" for k in keywords)
    + "\n"
)
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