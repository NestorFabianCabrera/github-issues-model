import requests
import csv
import time
import os
import sys

TOKEN = os.environ["GITHUB_TOKEN"]
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

REPOS = [
    "expressjs/express",
    "vercel/next.js",
    "vitejs/vite",
    "vuejs/vue",
    "sveltejs/svelte",
    "nestjs/nest",
    "pallets/flask",
    "pytest-dev/pytest",
    "prettier/prettier",
    "eslint/eslint",
    "webpack/webpack",
    "babel/babel",
    "denoland/deno",
    "typescriptlang/TypeScript",
    "redis/redis",
    "grafana/grafana",
    "prometheus/prometheus",
    "docker/compose",
    "hashicorp/terraform",
    "ansible/ansible",
]

PAGES_PER_REPO = 6
OUTPUT = "/data/github_issues_extra.csv"

FIELDS = [
    "repo", "issue_number", "title", "body",
    "labels", "has_assignee", "has_milestone",
    "comments", "created_at", "closed_at",
    "day_of_week", "hour_of_day",
]


def fetch_issues(repo, page):
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": "closed", "per_page": 100, "page": page}
    r = requests.get(url, headers=HEADERS, params=params)
    if r.status_code == 403:
        reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait = max(reset - int(time.time()), 1)
        print(f"  Rate limit, waiting {wait}s...")
        time.sleep(wait)
        return fetch_issues(repo, page)
    r.raise_for_status()
    return r.json()


def parse_issue(repo, issue):
    if issue.get("pull_request"):
        return None
    created = issue.get("created_at", "")
    closed = issue.get("closed_at", "")
    if not closed:
        return None
    labels = "|".join(l["name"] for l in issue.get("labels", []))
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        dow = dt.weekday()
        hour = dt.hour
    except Exception:
        dow = -1
        hour = -1

    return {
        "repo": repo,
        "issue_number": issue["number"],
        "title": (issue.get("title") or "").replace("\n", " ").replace("\r", ""),
        "body": (issue.get("body") or "")[:2000].replace("\n", " ").replace("\r", ""),
        "labels": labels,
        "has_assignee": 1 if issue.get("assignees") else 0,
        "has_milestone": 1 if issue.get("milestone") else 0,
        "comments": issue.get("comments", 0),
        "created_at": created,
        "closed_at": closed,
        "day_of_week": dow,
        "hour_of_day": hour,
    }


def main():
    total = 0
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for repo in REPOS:
            print(f"\n→ {repo}")
            repo_count = 0
            for page in range(1, PAGES_PER_REPO + 1):
                issues = fetch_issues(repo, page)
                if not issues:
                    break
                for issue in issues:
                    row = parse_issue(repo, issue)
                    if row:
                        writer.writerow(row)
                        repo_count += 1
                sys.stdout.write(f"  page {page}: {repo_count} issues\r")
                sys.stdout.flush()
                time.sleep(0.3)
            total += repo_count
            print(f"  {repo}: {repo_count} issues")

    print(f"\nTotal extra: {total} → {OUTPUT}")


if __name__ == "__main__":
    main()
