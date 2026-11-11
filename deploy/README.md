# EC2 배포 구성

강의실 지킴이(FastAPI + React)를 서울 리전 EC2 한 대에 Docker Compose로 운영하는 설계입니다. 아직 AWS 리소스는 만들지 않았습니다. 이 문서의 구성을 기준으로 다음 단계에서 GitHub Actions 배포 workflow(또는 AWS CLI 배포 스크립트)를 작성합니다.

## 왜 EC2인가

| 기준 | EC2 (선택) | ECS Fargate |
|---|---|---|
| 상태 저장 | SQLite·모델 레지스트리를 EBS 위 Docker volume에 그대로 보존 | 영속 저장에 EFS 필요, SQLite WAL·파일 잠금이 네트워크 파일시스템에서 불안정 |
| 단일 실행 보장 | 공유 시연 시계와 SQLite 서비스 잠금이 한 프로세스를 전제로 함. 한 대·한 worker로 자연스럽게 충족 | 태스크 수를 1로 고정하고 배포 중 중복 태스크(rolling)를 막는 설정 필요 |
| 학습 작업 | API가 띄우는 학습 서브프로세스가 같은 디스크의 가중치·보고서에 기록 | 태스크 재시작 시 진행 중 학습·임시 파일 유실 위험 |
| 비용·복잡도 | 인스턴스 1대, ALB 불필요 | ALB·EFS·태스크 정의·서비스 구성 추가 |

규모가 작고 상태가 한 곳에 모여 있어 EC2 단일 인스턴스가 가장 단순합니다. 단일 인스턴스이므로 배포·장애 때 짧은 중단이 생깁니다.

## 아키텍처

```text
브라우저 ──HTTP 80──▶ [EC2 · 보안 그룹: 80만 허용]
                        └ Docker Compose (deploy/compose.yaml)
                           ├ web : nginx (정적 React 빌드, :8080 → 호스트 80)
                           │        ├ /            → index.html·assets
                           │        ├ /api/*       → api:8000 (접두사 제거)
                           │        └ /openapi.json, /static/swagger/* → api:8000
                           └ api : FastAPI + PyTorch CPU (uvicorn 1 worker, 비루트 uid 10001)
                                    └ volumes: state(SQLite) · logs · weights · reports  ← EBS

GitHub Actions ──OIDC──▶ IAM 역할 ──▶ ECR push (태그 = Git SHA)
                                   └─▶ SSM Run Command ──▶ EC2: ECR pull → compose up → 헬스체크
관리자 ──SSM Session Manager──▶ EC2   (SSH 키·22번 포트 없음)
```

## 리소스

| 항목 | 구성 | 근거 |
|---|---|---|
| 리전 | `ap-northeast-2` (서울) | 사용자·팀 위치 |
| 네트워크 | 전용 VPC 1개, 퍼블릭 서브넷 1개, 인터넷 게이트웨이. NAT·ALB 없음 | 단일 인스턴스, 공인 IP 직접 접속 |
| EC2 | `t3.small` (2 vCPU, 2GiB), Amazon Linux 2023 **x86_64** | 로컬 측정: api 대기 220MiB·학습 중 275MiB, web 12MiB. 이미지를 x86_64로 빌드하므로 아키텍처 일치 |
| 스왑 | 2GiB swap 파일 | 학습·이미지 pull 때 메모리 여유 |
| EBS | gp3 30GiB, 암호화 | API 이미지 1.75GB + 이전 이미지 보존(롤백) + 볼륨·로그 |
| 고정 IP | Elastic IP 1개 | 재시작 후에도 같은 접속 주소 |
| 보안 그룹 | 인바운드 TCP 80만. 아웃바운드 443(ECR·SSM·패키지), 80 | 관리 포트 비공개 |
| 관리 접속 | SSM Session Manager | SSH 키 보관·22번 포트 개방 불필요 |
| 메타데이터 | IMDSv2 필수, hop limit 1 | 컨테이너가 인스턴스 역할 자격증명을 가져가지 못하게 함 |
| 이미지 저장소 | private ECR `classroom-occupancy/api`, `classroom-occupancy/web`. 태그 변경 불가, Git SHA 태그 | 커밋별 이미지 고정, 이전 SHA로 롤백 |

## 권한 (장기 Access Key 없음)

**EC2 인스턴스 역할**
- `AmazonSSMManagedInstanceCore`: SSM Agent 연결, Run Command·Session Manager.
- ECR 두 저장소의 pull 전용 권한(`ecr:GetAuthorizationToken`, `BatchGetImage`, `GetDownloadUrlForLayer`).

**GitHub Actions 배포 역할 (OIDC)**
- 신뢰 조건: `token.actions.githubusercontent.com`, `aud=sts.amazonaws.com`, `sub=repo:<owner>/<repo>:ref:refs/heads/main`.
- ECR 두 저장소에 push.
- 이 인스턴스 한 대에만 배포용 SSM 명령 실행과 결과 조회.
- IAM·EC2 생성/변경 권한 없음.

GitHub 저장소 설정:

| 종류 | 이름 | 값 |
|---|---|---|
| Secret | `AWS_ROLE_ARN` | 배포 역할 ARN |
| Secret | `AWS_INSTANCE_ID` | EC2 인스턴스 ID |
| Variable | `AWS_DEPLOY_ENABLED` | 리소스 준비 후 `true` |

OIDC를 쓸 수 없는 계정이라면 4단계에서 권한 있는 사용자의 AWS CLI 프로필로 같은 흐름을 실행하는 배포 스크립트를 사용합니다. 어느 경우에도 Access Key를 저장소나 workflow에 넣지 않습니다.

## 컨테이너 구성 (`deploy/`)

| 파일 | 역할 |
|---|---|
| `compose.yaml` | 운영 Compose. 소스 바인드 마운트·`--reload`·Vite 개발 서버 없이 이미지만 실행 |
| `Dockerfile.web` | Node로 Vite 정적 빌드 → `nginx-unprivileged`로 서빙 |
| `nginx.conf` | 정적 파일, `/api` 프록시, Swagger 경로, 보안 헤더, 요청 본문 64KiB 제한 |
| `Dockerfile.web.dockerignore` | 화면 빌드에 프론트엔드 소스와 nginx 설정만 전달 |

API 이미지는 기존 `backend/Dockerfile`을 그대로 씁니다. 운영 컨테이너는 비루트(uid 10001)로 실행되며, 읽기 전용 루트 파일시스템에 쓰기 가능한 volume만 붙입니다.

| Volume | 경로 | 내용 |
|---|---|---|
| `state` | `/app/state` | 강의실 SQLite, 공유 시연 위치 |
| `logs` | `/app/logs` | API·모델 실행 로그 |
| `weights` | `/app/models/weights` | 모델 가중치와 `current.json` |
| `reports` | `/app/reports` | 평가 보고서와 학습 작업 이력 |

`weights`·`reports`는 처음 만들어질 때 이미지에 든 v1~v7 가중치·보고서로 채워집니다. 그 뒤로는 서버에서 학습한 버전과 기본 모델 선택을 유지하기 위해 **이미지를 새로 배포해도 덮어쓰지 않습니다.** 저장소에 새 가중치를 커밋해 서버에 반영하려면 별도 동기화 절차가 필요합니다(4단계에서 결정).

로컬 검증:

```sh
BIND_ADDRESS=127.0.0.1 HTTP_PORT=8088 docker compose -f deploy/compose.yaml up -d --build --wait
curl http://127.0.0.1:8088/api/health
docker compose -f deploy/compose.yaml down -v   # 검증용 볼륨까지 삭제
```

## 배포 흐름 (4단계 구현 대상)

1. **검증** (모든 push·PR): 백엔드 단위 테스트, `docker compose -f deploy/compose.yaml config`, 두 이미지 빌드, Compose 기동 후 `/api/health`·`/api/classrooms` 확인.
2. **게시** (`main`만): OIDC로 역할 위임 → 두 이미지를 `<계정>.dkr.ecr.ap-northeast-2.amazonaws.com/classroom-occupancy/{api,web}:<SHA>`로 push.
3. **배포**: SSM Run Command로 서버의 배포 스크립트 실행.
   - ECR 로그인 → `/opt/classroom/.env`에 `API_IMAGE`·`WEB_IMAGE`를 새 SHA로 기록.
   - `docker compose pull` → `docker compose up -d --no-build --wait`.
   - `http://127.0.0.1/api/health`가 실패하면 직전 SHA로 되돌려 다시 기동.
   - 성공한 SHA를 `/opt/classroom/deployed-commit`에 기록.
4. **확인**: Actions 요약에 배포 SHA와 접속 주소를 남기고, 공인 IP로 화면·API를 확인(5단계).

서버 파일 배치: `/opt/classroom/compose.yaml`(이 디렉터리의 `compose.yaml`), `/opt/classroom/.env`(이미지 태그·포트, 비밀값 없음). 중복 배포는 `flock`으로 막고 workflow도 `concurrency`로 직렬화합니다.

## 초기 구축 순서

1. ECR 저장소 2개 생성(태그 변경 불가, 오래된 이미지 수명 주기 정책).
2. VPC·서브넷·인터넷 게이트웨이·보안 그룹(80 인바운드) 생성.
3. EC2 인스턴스 역할 생성 후 `t3.small` 시작: AL2023 x86_64, gp3 30GiB 암호화, IMDSv2 필수, Elastic IP 연결.
4. user-data: Docker·Compose 플러그인 설치, 2GiB swap, `/opt/classroom` 생성 및 `compose.yaml`·배포 스크립트 배치.
5. GitHub OIDC provider와 배포 역할 생성, 저장소 Secrets·Variable 등록.
6. 첫 배포 실행 후 `http://<Elastic IP>/` 확인.

반복 가능하도록 1~5는 CloudFormation 템플릿이나 AWS CLI 스크립트로 작성하고, 생성한 리소스 ID는 `deploy/deployment.json`에 기록합니다.

## 예상 비용 (서울, 월 730시간)

| 항목 | 계산 | USD |
|---|---|---:|
| EC2 t3.small | $0.026 × 730 | 18.98 |
| EBS gp3 30GiB | $0.0912 × 30 | 2.74 |
| 공인 IPv4 | $0.005 × 730 | 3.65 |
| ECR 저장 | 약 2~4GB × $0.10 | 0.2~0.4 |
| 합계 | | **약 $26 / 월** |

데이터 전송·세금 제외, 무료 티어를 가정하지 않은 값입니다. 인스턴스를 중지해도 EBS·Elastic IP 비용은 계속 나가므로 실습이 끝나면 EC2·EIP·EBS·ECR을 함께 정리해야 합니다.

## 운영 전 결정할 사항

- **공개 범위**: 화면과 API에 로그인이 없어 누구나 학습을 시작하거나 기본 모델을 바꿀 수 있습니다. 시연 기간에는 보안 그룹 80번 인바운드를 팀·학교 IP 대역으로 제한하는 것을 권장합니다.
- **HTTPS**: 도메인이 없어 HTTP로 공개합니다. 인증이나 개인정보를 다루기 전에 도메인과 TLS를 구성해야 합니다.
- **가중치 동기화**: 서버 volume과 저장소의 가중치를 어떻게 맞출지(서버 우선 유지 / 배포 때 새 버전만 추가) 4단계 전에 정합니다.
- **백업**: SQLite·가중치 volume 정기 백업(S3 등)은 아직 계획에 없습니다.
