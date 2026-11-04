#!/usr/bin/env bash
# EC2에서 지정한 Git 커밋을 빌드·기동하고, 헬스체크에 실패하면 직전 배포 커밋으로 되돌린다.
# GitHub Actions의 배포 전용 SSH 키는 authorized_keys의 command= 옵션으로 이 스크립트만 실행할 수 있다.
# 사용: deploy.sh <40자리 커밋 SHA>  (강제 명령으로 호출되면 SSH_ORIGINAL_COMMAND에서 SHA를 읽는다)
set -euo pipefail

main() {
    local target="${1:-${SSH_ORIGINAL_COMMAND:-}}"
    if [[ ! "$target" =~ ^[0-9a-f]{40}$ ]]; then
        echo "배포할 40자리 커밋 SHA가 필요합니다." >&2
        return 2
    fi

    local repo state_dir
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    state_dir="$HOME/classroom-deploy"
    mkdir -p "$state_dir"

    exec 9>"$state_dir/deploy.lock"
    if ! flock -n 9; then
        echo "다른 배포가 진행 중입니다." >&2
        return 1
    fi

    local previous
    previous="$(cat "$state_dir/deployed-commit" 2>/dev/null || git -C "$repo" rev-parse HEAD)"

    git -C "$repo" fetch --quiet origin
    if ! git -C "$repo" cat-file -e "$target^{commit}" 2>/dev/null; then
        echo "원격 저장소에서 커밋 $target 을 찾을 수 없습니다." >&2
        return 1
    fi

    echo "배포 시작: $previous → $target"
    git -C "$repo" checkout --quiet --detach "$target"
    if start_stack "$repo" && healthy "$repo"; then
        echo "$target" > "$state_dir/deployed-commit"
        echo "배포 완료: $target"
        return 0
    fi

    echo "배포 실패: 직전 커밋 $previous 으로 되돌립니다." >&2
    git -C "$repo" checkout --quiet --detach "$previous"
    if start_stack "$repo" && healthy "$repo"; then
        echo "롤백 완료: $previous" >&2
    else
        echo "롤백도 실패했습니다. 서버에서 docker compose 로그를 확인하세요." >&2
    fi
    return 1
}

start_stack() {
    docker compose -f "$1/deploy/compose.yaml" --env-file "$1/deploy/.env" \
        up -d --build --remove-orphans --wait --wait-timeout 300
}

healthy() {
    local port
    port="$(sed -n 's/^HTTP_PORT=//p' "$1/deploy/.env" 2>/dev/null | tail -1)"
    port="${port:-80}"
    curl -fsS -m 10 "http://127.0.0.1:$port/api/health" && echo
    curl -fsS -m 10 -o /dev/null "http://127.0.0.1:$port/api/classrooms"
    curl -fsS -m 10 -o /dev/null "http://127.0.0.1:$port/"
}

# 배포 중 이 파일이 새 커밋으로 바뀌어도 이미 읽은 함수만 실행하도록 같은 줄에서 종료한다.
main "$@"; exit $?
